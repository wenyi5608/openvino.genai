### 1. Prepare Python Virtual Environment for LLM Converting
   
``` bash
python3 -m venv ov-llm-bench-env
source ov-llm-bench-env/bin/activate
pip install --upgrade pip

git clone  https://github.com/wenyi5608/openvino.genai.git -b wenyi5608-stateful_token
cd openvino.genai/tools/llm_bench
pip install -r requirements_2024.5.txt  
```

> Note:
> For existing Python environments, run the following command to ensure that all dependencies are installed with the latest versions:  
> `pip install -U --upgrade-strategy eager -r requirements.txt`

#### (Optional) Hugging Face Login :

Login to Hugging Face if you want to use non-public models:

```bash
huggingface-cli login
```

### 2. Convert Model to OpenVINO IR Format
   
The `optimum-cli` tool simplifies converting Hugging Face models to OpenVINO IR format. 
- Detailed documentation can be found in the [Optimum-Intel documentation](https://huggingface.co/docs/optimum/main/en/intel/openvino/export). 
- To learn more about weight compression, see the [NNCF Weight Compression Guide](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html).
- For additional guidance on running inference with OpenVINO for LLMs, see the [OpenVINO LLM Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

**Usage:**

```bash
optimum-cli export openvino --model <MODEL_ID> --weight-format <PRECISION> <OUTPUT_DIR>

optimum-cli export openvino -h # For detailed information
```

* `--model <MODEL_ID>` : model_id for downloading from [huggngface_hub](https://huggingface.co/models) or path with directory where pytorch model located. 
* `--weight-format <PRECISION>` : precision for model conversion. Available options: `fp32, fp16, int8, int4, mxfp4`
* `<OUTPUT_DIR>`: output directory for saving generated OpenVINO model.

**NOTE:** 
- Models larger than 1 billion parameters are exported to the OpenVINO format with 8-bit weights by default. You can disable it with `--weight-format fp32`.

**Example:**
```bash
optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf --weight-format fp16 models/llama-2-7b-chat
```
**Resulting file structure:**

```console
    models
    └── llama-2-7b-chat
        ├── config.json
        ├── generation_config.json
        ├── openvino_detokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_model.bin
        ├── openvino_model.xml
        ├── openvino_tokenizer.bin
        ├── openvino_tokenizer.xml
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json
        └── tokenizer.model
```

> **Note:** If needed, You can install a specific OpenVINO version using pip:
> ``` bash
> # e.g. 
> pip install openvino==2024.4.0
> # Optional, install the openvino nightly package if needed.
> # OpenVINO nightly is pre-release software and has not undergone full release validation or qualification. 
> pip uninstall openvino
> pip install --upgrade --pre openvino openvino-tokenizers --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
> ```

### 2. Compress Model with BackupMode
```bash
python minicpm_compress.py -m /path/to/fp16/openvino_model.xml -o /path/to/outputdir
```

### 3. Compress Model with GPTQ
```bash
python minicpm_gptq.py -m /path/to/MiniCPM-1B_ov/openvino_model.xml -o /path/to/MiniCPM-1B_ov/gptq/  --token /path/to/MiniCPM-1B-sft-bf16/ --num_samples 10 --dataset /path/to/dataset.txt
```

## 4. Additional Resources

- **Error Troubleshooting:** Check the [NOTES.md](./doc/NOTES.md) for solutions to known issues.
- **Syntax and attributes of prompt file:** Refer to [PROMPT.md](./doc/PROMPT.md) for writing a prompt file.
