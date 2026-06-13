# CARLA Runtime Assets

`carla_start_runtime.sh` prefers these SimLingo-local runtime assets:

```text
runtime/carla_0.9.15.sif
runtime/nvidia-driver-libs/<driver-version>/
```

The SIF and driver-library bundle are intentionally ignored by Git because
they are large and cluster-specific.

Until those assets are placed here, the starter temporarily falls back to:

```text
/lab/haoq_lab/cse12312032/openvla/project/carla_0.9.15.sif
/lab/haoq_lab/cse12312032/openvla/nvidia-driver-libs/
```

Start a runtime from the SimLingo repository with:

```bash
cd /lab/haoq_lab/cse12312032/simlingo
sbatch --export=ALL,CARLA_MAP=Town10HD_Opt carla_start_runtime.sh
```

New connection files and server logs are written to:

```text
logs/carla_runtime/<timestamp>_job<job-id>/
```
