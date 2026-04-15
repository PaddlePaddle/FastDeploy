## 2024-04-15 - FastDeploy test suite
**Learning:** The fastdeploy codebase tests require a complex environment. Running `pytest tests/` directly fails with hundreds of import errors due to missing dependencies and environment setup specific to FastDeploy (such as PaddlePaddle and other ML packages).
**Action:** When working on this codebase, accept that local tests might fail unless running in a fully configured container or environment.
