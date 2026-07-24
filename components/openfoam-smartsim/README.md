# SmartSim-CSC OpenFOAM Integration

This component provides OpenFOAM v2412 applications and libraries that connect
OpenFOAM simulations to the SmartSim-CSC SmartRedis database.

## Currently verified

- OpenFOAM.com v2412
- GCC 15.2.0
- SmartRedis built from `components/smartredis`
- `foamSmartSimSvd`
- Serial OpenFOAM field transfer to SmartRedis
- Python retrieval of a `(400, 50)` pressure tensor

## Build

Load the SmartSim-CSC and OpenFOAM environments, then run
`scripts/openfoam/build-openfoam-v2412.sh`.

The executable is installed under the configured `FOAM_USER_APPBIN`.
