# Owner(s): ["module: onnx"]

from __future__ import annotations

import copy

import dataclasses
import io
import os
import warnings
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np

import onnxruntime
import pytorch_test_common
import torch
from torch.onnx import _constants, verification
from torch.onnx._internal import _beartype
from torch.types import Number

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, Callable]
_InputArgsType = Optional[
    Union[torch.Tensor, int, float, bool, Sequence[Any], Mapping[str, Any]]
]
_OutputsType = Sequence[_NumericType]

onnx_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    "repos",
    "onnx",
    "onnx",
    "backend",
    "test",
    "data",
)


pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")


pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")


def run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    options = verification.VerificationOptions()

    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    if hasattr(test_suite, "check_shape"):
        options.check_shape = test_suite.check_shape
    if hasattr(test_suite, "check_dtype"):
        options.check_dtype = test_suite.check_dtype

    names = {f.name for f in dataclasses.fields(options)}
    keywords_to_pop = []
    for k, v in kwargs.items():
        if k in names:
            setattr(options, k, v)
            keywords_to_pop.append(k)
    for k in keywords_to_pop:
        kwargs.pop(k)

    return verification.verify(*args, options=options, **kwargs)


def parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffix = "_".join(f"{k}_{v}" for k, v in input_dicts.items())
    return f"{cls.__name__}_{suffix}"


class _TestONNXRuntime(pytorch_test_common.ExportTestCase):
    opset_version = _constants.ONNX_DEFAULT_OPSET
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    is_script = False
    check_shape = True
    check_dtype = True

    def setUp(self):
        super().setUp()
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
        self.is_script_test_enabled = True

    # The exported ONNX model may have less inputs than the pytorch model because of const folding.
    # This mostly happens in unit test, where we widely use torch.size or torch.shape.
    # So the output is only dependent on the input shape, not value.
    # remained_onnx_input_idx is used to indicate which pytorch model input idx is remained in ONNX model.
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=torch.onnx.TrainingMode.EVAL,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        def _run_test(m, remained_onnx_input_idx, flatten=True, ignore_none=True):
            return run_model_test(
                self,
                m,
                input_args=input_args,
                input_kwargs=input_kwargs,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                additional_test_inputs=additional_test_inputs,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                remained_onnx_input_idx=remained_onnx_input_idx,
                flatten=flatten,
                ignore_none=ignore_none,
                verbose=verbose,
            )

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        if self.is_script_test_enabled and self.is_script:
            script_model = model if is_model_script else torch.jit.script(model)
            _run_test(
                script_model,
                scripting_remained_onnx_input_idx,
                flatten=False,
                ignore_none=False,
            )
        if not is_model_script and not self.is_script:
            _run_test(model, tracing_remained_onnx_input_idx)

    @_beartype.beartype
    def run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
        self,
        model: _ModelType,
        input_args: Sequence[_InputArgsType],
        input_kwargs: Optional[Mapping[str, _InputArgsType]] = None,
        rtol: float = 1e-3,
        atol: float = 1e-7,
        opset_version: int = 18,
        has_mutation: bool = False,
        additional_test_inputs: Optional[
            List[
                Union[
                    Tuple[Sequence[_InputArgsType], Mapping[str, _InputArgsType]],
                    Tuple[Sequence[_InputArgsType]],
                ]
            ]
        ] = None,
    ):
        """Compare the results of PyTorch model with exported ONNX model

        Args:
            model (_ModelType): PyTorch model
            input_args (Sequence[_InputArgsType]): torch input arguments
            input_kwargs (Mapping[str, _InputArgsType]): torch input kwargs
            rtol (float, optional): relative tolerance. Defaults to 1e-3.
            atol (float, optional): absolute tolerance. Defaults to 1e-7.
            opset_version (int, optional): ONNX opset version. Defaults to 18.
            has_mutation (bool, optional): Whether the model mutates its input or state.
                `mutation` as `True` incurs extra overhead of cloning the inputs and model.
                Defaults to False.
            additional_test_inputs: Test the models with another dataset input, which
                is designed for dynamic axes testing. Defaults to None. It's a list of
                different input sets in tuples. Inside tuple, the first element is a tuple
                of args, and the second element is a dict of kwargs. Remember to put comma
                even if the following element is not provided.
                For example,
                additional_test_inputs = [((args1, args2), {"kwargs":1}), ((args1,),), ((), {"kwargs":1})]

        """

        # avoid mutable data structure
        if input_kwargs is None:
            input_kwargs = {}

        if has_mutation:
            ref_model = _try_clone_model(model)
            ref_input_args, ref_input_kwargs = _try_clone_inputs(
                input_args, input_kwargs
            )
        else:
            ref_model = model
            ref_input_args = input_args
            ref_input_kwargs = input_kwargs

        # Feed args and kwargs into exporter.
        # Note that exporter should flatten kwargs into positional args the exported model;
        # since ONNX doesn't represent kwargs.
        export_output = torch.onnx.dynamo_export(
            ref_model,
            *ref_input_args,
            **ref_input_kwargs,
            export_options=torch.onnx.ExportOptions(
                opset_version=opset_version,
                op_level_debug=self.op_level_debug,
                dynamic_shapes=self.dynamic_shapes,
            ),
        )

        _compare_pytorch_onnx_with_ort(
            export_output,
            model,
            input_args,
            input_kwargs,
            atol,
            rtol,
            has_mutation=has_mutation,
        )
        # This confirms the exported mode accepts different input shapes
        # when dynamic shape is enabled.
        if additional_test_inputs and self.dynamic_shapes:
            for another_input in additional_test_inputs:
                if len(another_input) > 2:
                    raise ValueError(
                        f"test_inputs should only have tuple args and dictionary kwargs. But receives: {len(another_input)}"
                    )
                additional_input_args = another_input[0]
                additional_input_kwargs = (
                    another_input[1]
                    if len(another_input) == 2 and another_input[1] is not None
                    else {}
                )
                _compare_pytorch_onnx_with_ort(
                    export_output,
                    model,
                    additional_input_args,
                    additional_input_kwargs,
                    atol,
                    rtol,
                    has_mutation=has_mutation,
                )


@_beartype.beartype
def run_ort(
    onnx_model: Union[str, torch.onnx.ExportOutput],
    pytorch_inputs: Sequence[_InputArgsType],
) -> _OutputsType:
    """Run ORT on the given ONNX model and inputs

    Used in test_fx_to_onnx_with_onnxruntime.py

    Args:
        onnx_model (Union[str, torch.onnx.ExportOutput]): Converter ONNX model
        pytorch_inputs (Sequence[_InputArgsType]): The given torch inputs

    Raises:
        AssertionError: ONNX and PyTorch should have the same input sizes

    Returns:
        _OutputsType: ONNX model predictions
    """
    if isinstance(onnx_model, torch.onnx.ExportOutput):
        buffer = io.BytesIO()
        onnx_model.save(buffer)
        ort_model = buffer.getvalue()
    else:
        ort_model = onnx_model
    session = onnxruntime.InferenceSession(
        ort_model, providers=["CPUExecutionProvider"]
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]
    if len(input_names) != len(pytorch_inputs):
        raise AssertionError(
            f"Expected {len(input_names)} inputs, got {len(pytorch_inputs)}"
        )
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
    )


@_beartype.beartype
def _try_clone_model(model: _ModelType) -> _ModelType:
    """Used for preserving original model in case forward mutates model states."""
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification."
        )
        return model


@_beartype.beartype
def _try_clone_inputs(input_args, input_kwargs):
    ref_input_args = copy.deepcopy(input_args)
    ref_input_kwargs = copy.deepcopy(input_kwargs)
    return ref_input_args, ref_input_kwargs


@_beartype.beartype
def _compare_pytorch_onnx_with_ort(
    export_output: torch.onnx.ExportOutput,
    model: _ModelType,
    input_args: Sequence[_InputArgsType],
    input_kwargs: Mapping[str, _InputArgsType],
    atol: float,
    rtol: float,
    has_mutation: bool = False,
):
    if has_mutation:
        ref_model = _try_clone_model(model)
        ref_input_args, ref_input_kwargs = _try_clone_inputs(input_args, input_kwargs)
    else:
        ref_model = model
        ref_input_args = input_args
        ref_input_kwargs = input_kwargs

    # Format original model inputs into the format expected by exported ONNX model.
    onnx_format_args = export_output.adapt_torch_inputs_to_onnx(
        *input_args, **input_kwargs
    )

    ref_outputs = export_output.adapt_torch_outputs_to_onnx(
        ref_model(*ref_input_args, **ref_input_kwargs)
    )
    ort_outputs = run_ort(export_output, onnx_format_args)
    if len(ref_outputs) != len(ort_outputs):
        raise AssertionError(
            f"Expected {len(ref_outputs)} outputs, got {len(ort_outputs)}"
        )
    for ref_output, ort_output in zip(ref_outputs, ort_outputs):
        torch.testing.assert_close(
            ref_output, torch.tensor(ort_output), rtol=rtol, atol=atol
        )
