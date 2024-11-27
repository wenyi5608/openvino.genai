import argparse
from pathlib import Path

import openvino as ov

import logging as log
import numpy as np
import nncf
from nncf.scopes import IgnoredScope

from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops
from nncf import BackupMode

class InsertSlice(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Result")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            print("root: ", root)
            if root is None:
                return False
            root_output = matcher.get_match_value()
            print("root_output", root_output)
            root_name = root.get_friendly_name()
            if (len(root.get_output_partial_shape(0)) == 3):
                print(f"Find target root node name: {root_name}")
                parent = root.input_value(0).get_node()
                print(f"Find target parent node name: {parent.get_friendly_name()}")
                grand_parent = parent.input_value(0).get_node()
                print(f"Find grandparent node name: {grand_parent.get_friendly_name()}")
                grand_parent_output = parent.input(0).get_source_output()
                print("grand_parent_output: ", grand_parent_output)
                consumers = grand_parent_output.get_target_inputs()
                
                print(f"consumers: {consumers}")
                print("Original reshape node output shape:", grand_parent_output.get_partial_shape())
                dims = grand_parent_output.get_partial_shape().get_min_shape()[2]
                print("grand_parent_output dims : ", dims)
                start = np.array([0, -1, 0], dtype=np.int32)
                stop = np.array([1, -2, dims], dtype=np.int32)
                step = np.array([1, -1, 1], dtype=np.int32)
                axes = np.array([0, 1, 2], dtype=np.int32)
                slice = ops.slice(grand_parent, start, stop, step, axes, name="inserted_slice")
                print("After insert slice node, output shape:", slice.output(0).get_partial_shape())

                for consumer in consumers:
                    consumer.replace_source_output(slice.output(0))
                self.model_changed = True
                # Use new operation for additional matching
                self.register_new_node(slice)
                                
                return True

        self.register_matcher(Matcher(param,"InsertSlice"), callback)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Export minicpm-1b Model to IR", add_help=True)
    parser.add_argument("-m", "--model_file", required=True, help="OpenVINO model for loading")
    parser.add_argument("-o", "--output_dir", required=True, help="output directory for saving model")

    args = parser.parse_args()
    model_path = args.output_dir

    LLM_MODEL_OV = Path(args.model_file)
    LLM_MODEL_OV_INT4 = Path(f"{model_path}/openvino_model.xml")
    LLM_MODEL_OV_INT4_REDUCE_LOGITS = Path(f"{model_path}/modified_openvino_model.xml")
    
    if not LLM_MODEL_OV_INT4.exists() and LLM_MODEL_OV.exists():
        compression_configuration = {
            "mode": nncf.CompressWeightsMode.INT4_SYM,
            "group_size": 32
            }
        core = ov.Core()
        print("LLM model_ov", LLM_MODEL_OV)
        ov_model = core.read_model(LLM_MODEL_OV)
        ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration, backup_mode=BackupMode.NONE) #, ignored_scope=IgnoredScope(names=["aten::to/Convert"]))
        ov.save_model(ov_compressed_model, LLM_MODEL_OV_INT4)
    
    if not LLM_MODEL_OV_INT4_REDUCE_LOGITS.exists() and LLM_MODEL_OV_INT4.exists():
        core = ov.Core()
        ov_model = core.read_model(LLM_MODEL_OV_INT4)
        manager = Manager()
        manager.register_pass(InsertSlice())
        manager.run_passes(ov_model)
        ov.save_model(ov_model, LLM_MODEL_OV_INT4_REDUCE_LOGITS)

