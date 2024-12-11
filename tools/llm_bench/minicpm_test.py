import argparse

from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser("minicpm-1b Model test ", add_help=True)
    parser.add_argument("-m", "--model_id", required=True, help="OpenVINO model for loading")
    parser.add_argument("-d", "--device", default='cpu', help='inference device')

    args = parser.parse_args()
    model_dir = args.model_id

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device=args.device,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
    )

    test_string = "2 + 2 ="
    input_tokens = tok(test_string, return_tensors="pt")
    answer = ov_model.generate(**input_tokens, max_new_tokens=10)
    print(tok.batch_decode(answer, skip_special_tokens=True)[0])
