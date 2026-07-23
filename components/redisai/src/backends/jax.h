#pragma once

#include "config/config.h"
#include "execution/execution_contexts/execution_ctx.h"
#include "redis_ai_objects/err.h"
#include "redis_ai_objects/model.h"

int RAI_InitBackendJAX(int (*get_api_fn)(const char *, void **));

RAI_Model *RAI_ModelCreateJAX(
    RAI_Backend backend,
    const char *devicestr,
    RAI_ModelOpts opts,
    const char *modeldef,
    size_t modellen,
    RAI_Error *error
);

void RAI_ModelFreeJAX(
    RAI_Model *model,
    RAI_Error *error
);

int RAI_ModelRunJAX(
    RAI_Model *model,
    RAI_ExecutionCtx **ectxs,
    RAI_Error *error
);

int RAI_ModelSerializeJAX(
    RAI_Model *model,
    char **buffer,
    size_t *len,
    RAI_Error *error
);

const char *RAI_GetBackendVersionJAX(void);
