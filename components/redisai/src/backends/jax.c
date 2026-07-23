#define REDISMODULE_MAIN
#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "backends/util.h"
#include "backends/jax.h"
#include "backends/backends_api.h"
#include "redismodule.h"
#include "util/arr.h"

int RAI_InitBackendJAX(int (*get_api_fn)(const char *, void **)) {
    get_api_fn("RedisModule_Alloc", (void **)&RedisModule_Alloc);
    get_api_fn("RedisModule_Calloc", (void **)&RedisModule_Calloc);
    get_api_fn("RedisModule_Free", (void **)&RedisModule_Free);
    get_api_fn("RedisModule_Strdup", (void **)&RedisModule_Strdup);

    get_api_fn("RedisAI_TensorCreateFromDLTensor",
               (void **)&RedisAI_TensorCreateFromDLTensor);
    get_api_fn("RedisAI_TensorGetDLTensor",
               (void **)&RedisAI_TensorGetDLTensor);
    get_api_fn("RedisAI_TensorGetShallowCopy",
               (void **)&RedisAI_TensorGetShallowCopy);
    get_api_fn("RedisAI_TensorFree",
               (void **)&RedisAI_TensorFree);

    return REDISMODULE_OK;
}

RAI_Model *RAI_ModelCreateJAX(
    RAI_Backend backend,
    const char *devicestr,
    RAI_ModelOpts opts,
    const char *modeldef,
    size_t modellen,
    RAI_Error *error
) {
    RAI_Device device;
    int64_t device_id;

    if (!parseDeviceStr(devicestr, &device, &device_id)) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, "ERR unsupported JAX device");
        return NULL;
    }

    if (device != RAI_DEVICE_CPU && device != RAI_DEVICE_GPU) {
        RAI_SetError(error, RAI_EMODELCONFIGURE, "ERR unsupported JAX device");
        return NULL;
    }

    if (modeldef == NULL || modellen == 0) {
        RAI_SetError(error, RAI_EMODELCREATE, "ERR empty JAX model artifact");
        return NULL;
    }

    char *data = RedisModule_Alloc(modellen);
    if (data == NULL) {
        RAI_SetError(error, RAI_EMODELCREATE, "ERR unable to allocate JAX model data");
        return NULL;
    }
    memcpy(data, modeldef, modellen);

    char **inputs = array_new(char *, 1);
    char **outputs = array_new(char *, 1);

    inputs = array_append(inputs, RedisModule_Strdup(""));
    outputs = array_append(outputs, RedisModule_Strdup(""));

    RAI_Model *model = RedisModule_Calloc(1, sizeof(*model));
    if (model == NULL) {
        RedisModule_Free(inputs[0]);
        RedisModule_Free(outputs[0]);
        array_free(inputs);
        array_free(outputs);
        RedisModule_Free(data);
        RAI_SetError(error, RAI_EMODELCREATE, "ERR unable to allocate JAX model");
        return NULL;
    }

    /*
     * The model blob is retained unchanged. A persistent worker/session handle
     * will later be stored in model->session.
     */
    model->model = data;
    model->session = NULL;
    model->backend = backend;
    model->devicestr = RedisModule_Strdup(devicestr);
    model->opts = opts;

    model->inputs = inputs;
    model->ninputs = 1;
    model->outputs = outputs;
    model->noutputs = 1;

    model->refCount = 1;
    model->data = data;
    model->datalen = (long long)modellen;

    return model;
}

void RAI_ModelFreeJAX(RAI_Model *model, RAI_Error *error) {
    (void)error;

    if (model->session != NULL) {
        /*
         * Worker-side model/session destruction will be added with the IPC
         * implementation.
         */
        model->session = NULL;
    }

    if (model->devicestr != NULL) {
        RedisModule_Free(model->devicestr);
        model->devicestr = NULL;
    }

    if (model->inputs != NULL) {
        for (size_t i = 0; i < model->ninputs; ++i) {
            RedisModule_Free(model->inputs[i]);
        }
        array_free(model->inputs);
        model->inputs = NULL;
    }

    if (model->outputs != NULL) {
        for (size_t i = 0; i < model->noutputs; ++i) {
            RedisModule_Free(model->outputs[i]);
        }
        array_free(model->outputs);
        model->outputs = NULL;
    }

    if (model->data != NULL) {
        RedisModule_Free(model->data);
        model->data = NULL;
    }

    model->model = NULL;
}

static int jax_send_all(int fd, const char *buffer, size_t length) {
    size_t sent = 0;

    while (sent < length) {
        ssize_t result = send(fd, buffer + sent, length - sent, 0);

        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            return REDISMODULE_ERR;
        }

        if (result == 0) {
            return REDISMODULE_ERR;
        }

        sent += (size_t)result;
    }

    return REDISMODULE_OK;
}

static int jax_recv_line(int fd, char *buffer, size_t capacity) {
    size_t used = 0;

    while (used + 1 < capacity) {
        ssize_t result = recv(fd, buffer + used, 1, 0);

        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            return REDISMODULE_ERR;
        }

        if (result == 0) {
            break;
        }

        if (buffer[used] == '\n') {
            buffer[used] = '\0';
            return REDISMODULE_OK;
        }

        used++;
    }

    buffer[used] = '\0';
    return used > 0 ? REDISMODULE_OK : REDISMODULE_ERR;
}

int RAI_ModelRunJAX(
    RAI_Model *model,
    RAI_ExecutionCtx **ectxs,
    RAI_Error *error
) {
    if (model == NULL || model->data == NULL || model->datalen <= 0) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR JAX model artifact is unavailable");
        return REDISMODULE_ERR;
    }

    if (array_len(ectxs) != 1) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR JAX PoC currently supports one execution batch");
        return REDISMODULE_ERR;
    }

    RAI_ExecutionCtx *ectx = ectxs[0];

    if (RAI_ExecutionCtx_NumInputs(ectx) != 1 ||
        RAI_ExecutionCtx_NumOutputs(ectx) != 1) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR JAX PoC currently supports one input and one output");
        return REDISMODULE_ERR;
    }

    RAI_Tensor *input = RAI_ExecutionCtx_GetInput(ectx, 0);
    DLDataType input_dtype = RAI_TensorDataType(input);

    if (input_dtype.code != kDLFloat ||
        input_dtype.bits != 32 ||
        input_dtype.lanes != 1) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR JAX PoC currently supports FLOAT tensors only");
        return REDISMODULE_ERR;
    }

    int ndim = RAI_TensorNumDims(input);
    size_t length = RAI_TensorLength(input);
    float *input_data = (float *)RAI_TensorData(input);

    const unsigned char *model_data =
        (const unsigned char *)model->data;
    size_t model_length = (size_t)model->datalen;

    size_t request_capacity =
        512 + model_length * 2 + (size_t)ndim * 32 + length * 32;
    char *request = RedisModule_Alloc(request_capacity);

    if (request == NULL) {
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to allocate JAX request");
        return REDISMODULE_ERR;
    }

    size_t offset = 0;

    offset += (size_t)snprintf(
        request + offset,
        request_capacity - offset,
        "{\"model\":\""
    );

    for (size_t i = 0; i < model_length; ++i) {
        offset += (size_t)snprintf(
            request + offset,
            request_capacity - offset,
            "%02x",
            model_data[i]
        );
    }

    offset += (size_t)snprintf(
        request + offset,
        request_capacity - offset,
        "\",\"dtype\":\"float32\",\"shape\":["
    );

    for (int i = 0; i < ndim; ++i) {
        offset += (size_t)snprintf(
            request + offset,
            request_capacity - offset,
            "%s%lld",
            i == 0 ? "" : ",",
            RAI_TensorDim(input, i)
        );
    }

    offset += (size_t)snprintf(
        request + offset,
        request_capacity - offset,
        "],\"data\":["
    );

    for (size_t i = 0; i < length; ++i) {
        offset += (size_t)snprintf(
            request + offset,
            request_capacity - offset,
            "%s%.9g",
            i == 0 ? "" : ",",
            input_data[i]
        );
    }

    offset += (size_t)snprintf(
        request + offset,
        request_capacity - offset,
        "]}\n"
    );

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);

    if (fd < 0) {
        RedisModule_Free(request);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to create JAX worker socket");
        return REDISMODULE_ERR;
    }

    struct sockaddr_un address = {0};
    address.sun_family = AF_UNIX;

    const char *socket_path = getenv("SMARTSIM_JAX_SOCKET");
    if (socket_path == NULL || socket_path[0] == '\0') {
        socket_path = "/tmp/redisai-jax.sock";
    }

    if (strlen(socket_path) >= sizeof(address.sun_path)) {
        close(fd);
        RedisModule_Free(request);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR JAX worker socket path is too long");
        return REDISMODULE_ERR;
    }

    strcpy(address.sun_path, socket_path);

    if (connect(fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        close(fd);
        RedisModule_Free(request);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to connect to JAX worker");
        return REDISMODULE_ERR;
    }

    if (jax_send_all(fd, request, offset) != REDISMODULE_OK) {
        close(fd);
        RedisModule_Free(request);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to send request to JAX worker");
        return REDISMODULE_ERR;
    }

    RedisModule_Free(request);

    size_t response_capacity = 256 + length * 32;
    char *response = RedisModule_Alloc(response_capacity);

    if (response == NULL) {
        close(fd);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to allocate JAX response");
        return REDISMODULE_ERR;
    }

    int recv_status = jax_recv_line(fd, response, response_capacity);
    close(fd);

    if (recv_status != REDISMODULE_OK) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to receive response from JAX worker");
        return REDISMODULE_ERR;
    }

    if (strstr(response, "\"ok\": true") == NULL) {
        RAI_SetError(error, RAI_EMODELRUN, response);
        RedisModule_Free(response);
        return REDISMODULE_ERR;
    }

    char *dtype_start = strstr(response, "\"dtype\": \"");
    char *shape_start = strstr(response, "\"shape\": [");
    char *data_start = strstr(response, "\"data\": [");

    if (dtype_start == NULL || shape_start == NULL || data_start == NULL) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR invalid response from JAX worker");
        return REDISMODULE_ERR;
    }

    dtype_start += strlen("\"dtype\": \"");
    shape_start += strlen("\"shape\": [");
    data_start += strlen("\"data\": [");

    char *dtype_end = strchr(dtype_start, '"');

    if (dtype_end == NULL || dtype_end == dtype_start) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR invalid output dtype from JAX worker");
        return REDISMODULE_ERR;
    }

    size_t dtype_length = (size_t)(dtype_end - dtype_start);

    if (dtype_length >= 16) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR invalid output dtype from JAX worker");
        return REDISMODULE_ERR;
    }

    char worker_dtype[16];
    memcpy(worker_dtype, dtype_start, dtype_length);
    worker_dtype[dtype_length] = '\0';

    DLDataType output_dtype;
    size_t element_size = 0;
    int output_kind = -1;

    if (strcmp(worker_dtype, "float32") == 0) {
        output_dtype =
            (DLDataType){.code = kDLFloat, .bits = 32, .lanes = 1};
        element_size = sizeof(float);
        output_kind = 0;
    } else if (strcmp(worker_dtype, "float64") == 0) {
        output_dtype =
            (DLDataType){.code = kDLFloat, .bits = 64, .lanes = 1};
        element_size = sizeof(double);
        output_kind = 1;
    } else if (strcmp(worker_dtype, "int32") == 0) {
        output_dtype =
            (DLDataType){.code = kDLInt, .bits = 32, .lanes = 1};
        element_size = sizeof(int32_t);
        output_kind = 2;
    } else if (strcmp(worker_dtype, "int64") == 0) {
        output_dtype =
            (DLDataType){.code = kDLInt, .bits = 64, .lanes = 1};
        element_size = sizeof(int64_t);
        output_kind = 3;
    } else {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unsupported output dtype from JAX worker");
        return REDISMODULE_ERR;
    }

    char *shape_end = strchr(shape_start, ']');

    if (shape_end == NULL) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR invalid output shape from JAX worker");
        return REDISMODULE_ERR;
    }

    int output_ndim = 0;
    char *shape_cursor = shape_start;

    while (shape_cursor < shape_end) {
        while (shape_cursor < shape_end &&
               (*shape_cursor == ' ' || *shape_cursor == ',')) {
            shape_cursor++;
        }

        if (shape_cursor >= shape_end) {
            break;
        }

        char *end = NULL;
        strtoull(shape_cursor, &end, 10);

        if (end == shape_cursor || end > shape_end) {
            RedisModule_Free(response);
            RAI_SetError(error, RAI_EMODELRUN,
                         "ERR invalid output shape from JAX worker");
            return REDISMODULE_ERR;
        }

        output_ndim++;
        shape_cursor = end;
    }

    if (output_ndim <= 0) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR scalar JAX outputs are not supported yet");
        return REDISMODULE_ERR;
    }

    size_t *output_dims =
        RedisModule_Alloc((size_t)output_ndim * sizeof(size_t));

    if (output_dims == NULL) {
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to allocate JAX output shape");
        return REDISMODULE_ERR;
    }

    size_t output_length = 1;
    shape_cursor = shape_start;

    for (int i = 0; i < output_ndim; ++i) {
        while (shape_cursor < shape_end &&
               (*shape_cursor == ' ' || *shape_cursor == ',')) {
            shape_cursor++;
        }

        char *end = NULL;
        unsigned long long dim_value = strtoull(shape_cursor, &end, 10);

        if (end == shape_cursor || end > shape_end ||
            dim_value > (unsigned long long)((size_t)-1)) {
            RedisModule_Free(output_dims);
            RedisModule_Free(response);
            RAI_SetError(error, RAI_EMODELRUN,
                         "ERR invalid output shape from JAX worker");
            return REDISMODULE_ERR;
        }

        output_dims[i] = (size_t)dim_value;

        if (output_dims[i] != 0 &&
            output_length > ((size_t)-1) / output_dims[i]) {
            RedisModule_Free(output_dims);
            RedisModule_Free(response);
            RAI_SetError(error, RAI_EMODELRUN,
                         "ERR JAX output tensor is too large");
            return REDISMODULE_ERR;
        }

        output_length *= output_dims[i];
        shape_cursor = end;
    }

    if (output_length != 0 &&
        element_size > ((size_t)-1) / output_length) {
        RedisModule_Free(output_dims);
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR JAX output tensor is too large");
        return REDISMODULE_ERR;
    }

    size_t output_nbytes = output_length * element_size;
    void *output_data = RedisModule_Alloc(output_nbytes);

    if (output_data == NULL) {
        RedisModule_Free(output_dims);
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR unable to allocate JAX output");
        return REDISMODULE_ERR;
    }

    char *cursor = data_start;

    for (size_t i = 0; i < output_length; ++i) {
        char *end = NULL;
        errno = 0;

        if (output_kind == 0) {
            float value = strtof(cursor, &end);

            if (end != cursor && errno != ERANGE) {
                ((float *)output_data)[i] = value;
            }
        } else if (output_kind == 1) {
            double value = strtod(cursor, &end);

            if (end != cursor && errno != ERANGE) {
                ((double *)output_data)[i] = value;
            }
        } else if (output_kind == 2) {
            long value = strtol(cursor, &end, 10);

            if (end != cursor &&
                errno != ERANGE &&
                value >= INT32_MIN &&
                value <= INT32_MAX) {
                ((int32_t *)output_data)[i] = (int32_t)value;
            } else {
                end = cursor;
            }
        } else {
            long long value = strtoll(cursor, &end, 10);

            if (end != cursor && errno != ERANGE) {
                ((int64_t *)output_data)[i] = (int64_t)value;
            }
        }

        if (end == cursor || errno == ERANGE) {
            RedisModule_Free(output_data);
            RedisModule_Free(output_dims);
            RedisModule_Free(response);
            RAI_SetError(error, RAI_EMODELRUN,
                         "ERR invalid output data from JAX worker");
            return REDISMODULE_ERR;
        }

        cursor = end;

        while (*cursor == ' ' || *cursor == ',') {
            cursor++;
        }
    }

    if (*cursor != ']') {
        RedisModule_Free(output_data);
        RedisModule_Free(output_dims);
        RedisModule_Free(response);
        RAI_SetError(error, RAI_EMODELRUN,
                     "ERR output data count does not match output shape");
        return REDISMODULE_ERR;
    }

    RAI_Tensor *output = RAI_TensorCreateFromBlob(
        output_dtype,
        output_dims,
        output_ndim,
        (const char *)output_data,
        output_nbytes,
        error
    );

    RedisModule_Free(output_data);
    RedisModule_Free(output_dims);
    RedisModule_Free(response);

    if (output == NULL) {
        return REDISMODULE_ERR;
    }

    RAI_ExecutionCtx_SetOutput(ectx, output, 0);
    return REDISMODULE_OK;
}

int RAI_ModelSerializeJAX(
    RAI_Model *model,
    char **buffer,
    size_t *len,
    RAI_Error *error
) {
    if (model == NULL || model->data == NULL || model->datalen <= 0) {
        RAI_SetError(error, RAI_EMODELSERIALIZE, "ERR invalid JAX model data");
        return REDISMODULE_ERR;
    }

    *buffer = RedisModule_Alloc((size_t)model->datalen);
    if (*buffer == NULL) {
        RAI_SetError(error, RAI_EMODELSERIALIZE, "ERR unable to serialize JAX model");
        return REDISMODULE_ERR;
    }

    memcpy(*buffer, model->data, (size_t)model->datalen);
    *len = (size_t)model->datalen;

    return REDISMODULE_OK;
}

const char *RAI_GetBackendVersionJAX(void) {
    return "JAX-sidecar-poc-0.1";
}
