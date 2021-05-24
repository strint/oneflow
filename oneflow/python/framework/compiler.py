"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

from contextlib import contextmanager

import inspect
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.push_util as push_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.typing as oft
import oneflow.python.framework.typing_util as oft_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.ops as ops
import typing
import oneflow
import oneflow_api
import inspect


def Compile(session, function_desc, config_proto):
    # note(strint): lazy使用的
    #   一个新Job的创建
    with InterpretScope(session, function_desc, config_proto):
        # note(strint): 构建了job
        _CompileJob(session, function_desc)
        # note(strint): Complete中做了逻辑图各个pass的优化
        oneflow_api.CurJobBuildAndInferCtx_Complete()


def EagerRun(session, function_desc, config_proto, args):
    # note(strint): eager使用的
    with InterpretScope(session, function_desc, config_proto):
        ret = _InterpretGlobalFunction(function_desc, args)
        # note(strint): Complete中做了逻辑图优化
        oneflow_api.CurJobBuildAndInferCtx_Complete()
        session_ctx.GetDefaultSession().UpdateInfo4InterfaceOp()
    return ret


@contextmanager
def InterpretScope(session, function_desc, config_proto):
    job_conf = function_desc.job_config_proto
    job_conf.set_job_name(function_desc.job_func.__name__)
    placement_scope = function_desc.function_attribute.default_placement_scope
    if placement_scope is None:
        tag_and_dev_ids = placement_util.GetDefaultMachineDeviceIds(session.resource)
        hierarchy = None
    else:
        assert isinstance(placement_scope, placement_ctx.EmptyPlacementScope)
        tag_and_dev_ids = (
            placement_scope.device_tag,
            placement_scope.machine_device_ids,
        )
        hierarchy = placement_scope.hierarchy

    distribute_strategy = function_desc.function_attribute.default_distribute_strategy
    if distribute_strategy is None:
        distribute_strategy = distribute_util.DistributeConsistentStrategy()
    is_mirrored = isinstance(
        distribute_strategy, distribute_util.DistributeMirroredStrategy
    )
    assert isinstance(hierarchy, (list, tuple)) or hierarchy is None
    if hierarchy is not None:
        hierarchy = oneflow_api.Size(tuple(hierarchy))
    scope = scope_util.MakeInitialScope(
        job_conf, *tag_and_dev_ids, hierarchy, is_mirrored
    )
    # note(strint): 这里利用yield构造了一个contextmanager
    # note(strint): 打开了JobBuildAndInferCtx
    with _JobBuildAndInferCtx(job_conf.job_name()), distribute_strategy:
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
        # note(strint): 设置了当前mode为GLOBAL_MODE，表示进入了一个global_funciton
        with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
            with scope_util.ScopeContext(scope):
                # note(strint): InterpretScope的with会执行到这里然后yield，这里就建立了三个with context
                yield
                # note(strint): InterpretScope结束时，接着执行这里，这样就依次释放掉三个with context


def _SessionInitialScope(session, scope):
    job_name = scope.job_desc_symbol.data.job_name
    session.InitNoneScope(job_name)
    return session.NewCurrentScope(scope)


# note(strint): 编译job func
#   创建job输入输出的blob def
#   运行func进行把operator创建和infer好，并加入到当前Job中
def _CompileJob(session, function_desc):
    func = function_desc.job_func
    # note(strint): inspect从job_func获取的signature
    parameters = func.__oneflow_function_signature__.parameters
    # note(strint): 给job_fuc增加了__oneflow_input_blob_defs__
    #         根据job_func的输入参数列宾创建的input blob defs
    if len(parameters) == 0:
        # note(strint): job_func无参数
        func.__oneflow_input_blob_defs__ = ()
    elif all(p.annotation is inspect._empty for _, p in parameters.items()):
        func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    elif all(p.annotation is not inspect._empty for _, p in parameters.items()):
        func.__oneflow_input_blob_defs__ = _MakeInputBlobDefFromParameterSignature(
            parameters
        )
    else:
        raise NotImplementedError(
            "All parameters of global function should be annotated"
        )
    # note(strint): 根据input_blob_defs创建input blobs
    inputs = _RecursiveMakeInputBlobs(func.__oneflow_input_blob_defs__)
    # note(strint): 调用job_func，输入inputs blobs，返回loss blob
    #         里面调用的是OpBuilder，创建了OperatorConf，然后用 InferAndTryRun 方法加入了当前的Job中
    #         构建了logical graph
    ret = func(*inputs)

    return_annotation = func.__oneflow_function_signature__.return_annotation
    oft_util.CheckReturnByAnnotation(func.__name__, ret, return_annotation)
    # note(strint): 创建job_func返回blob,记录在out_put_remote_blobs
    func.__oneflow_output_remote_blobs__ = _RecursiveMakeRetRemoteBlobs(
        ret, allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op
    )
    session.StashJob(func.__name__)


def _InterpretGlobalFunction(function_desc, args):
    func = function_desc.job_func
    parameters = func.__oneflow_function_signature__.parameters
    if len(parameters) == 0:
        func.__oneflow_input_blob_defs__ = ()
    elif all(p.annotation is inspect._empty for _, p in parameters.items()):
        func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    elif all(p.annotation is not inspect._empty for _, p in parameters.items()):
        func.__oneflow_input_blob_defs__ = _MakeInputBlobDefFromParameterSignature(
            parameters
        )
    else:
        raise NotImplementedError(
            "All parameters of global function should be annotated"
        )
    inputs = push_util.MakeEagerInputBlobs(func.__oneflow_input_blob_defs__, args)
    ret = func(*inputs)
    return_annotation = func.__oneflow_function_signature__.return_annotation
    oft_util.CheckReturnByAnnotation(func.__name__, ret, return_annotation)
    return _RecursiveMakeRetRemoteBlobs(
        ret, allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op
    )


@contextmanager
def _JobBuildAndInferCtx(job_name):
    # note(strint): 开启一个job的创建
    c_api_util.JobBuildAndInferCtx_Open(job_name)
    try:
        yield
    finally:
        # note(strint): 结束一个job的创建
        oneflow_api.JobBuildAndInferCtx_Close()


def _GetArgDefault(func):
    if hasattr(func, "__oneflow_arg_default__"):
        return func.__oneflow_arg_default__
    return _CloneArgBlobDef(func_inspect_util.GetArgDefaults(func))


def _CloneArgBlobDef(args):
    if isinstance(args, input_blob_util.ArgBlobDef):
        return args.Clone()
    if isinstance(args, (tuple, list)):
        return type(args)(_CloneArgBlobDef(x) for x in args)
    if isinstance(args, dict):
        return {k: _CloneArgBlobDef(v) for k, v in args}
    raise NotImplementedError(
        "oneflow.global_function only accepts nested input blob defs"
    )


def _RecursiveMakeInputBlobs(input_blob_def):
    if isinstance(input_blob_def, input_blob_util.ArgBlobDef):
        return ops.InputOpByArgBlobDef(input_blob_def)
    if isinstance(input_blob_def, (tuple, list)):
        return type(input_blob_def)(_RecursiveMakeInputBlobs(x) for x in input_blob_def)
    if isinstance(input_blob_def, dict):
        return {k: _RecursiveMakeInputBlobs(v) for k, v in input_blob_def.items()}
    raise NotImplementedError(
        "oneflow.global_function accepts "
        + "ArgBlobDefs or list/tuple/dict nested ArgBlobDefs as argument"
    )


def _MakeInputBlobDefFromParameterSignature(parameters):
    def CheckAndRecusiveMake(p):
        return _RecusiveMakeInputBlobDef(p.annotation)

    return tuple(CheckAndRecusiveMake(p) for _, p in parameters.items())


def _RecusiveMakeInputBlobDef(cls):
    if oft.OriginFrom(cls, oft.OneflowNumpyDef):
        return cls.NewInputBlobDef()
    elif oft.OriginFrom(cls, typing.Tuple):
        return tuple(_RecusiveMakeInputBlobDef(a) for a in cls.__args__)
    else:
        raise NotImplementedError(
            ("\nannotation %s" % cls)
            + "not supported"
            + "\nonly support oneflow.typing.Numpy.Placeholder, "
            "oneflow.typing.ListNumpy.Placeholder"
        )


def _RecursiveMakeRetRemoteBlobs(remote_blobs, **kwarg):
    if remote_blobs is None:
        return None
    if isinstance(remote_blobs, oneflow_api.BlobDesc):
        return ops.ReturnRemoteBlob(remote_blobs, **kwarg)
    if isinstance(remote_blobs, (tuple, list)):
        return type(remote_blobs)(
            _RecursiveMakeRetRemoteBlobs(x, **kwarg) for x in remote_blobs
        )
    if isinstance(remote_blobs, dict):
        return {
            k: _RecursiveMakeRetRemoteBlobs(v, **kwarg) for k, v in remote_blobs.items()
        }
    raise NotImplementedError(
        "oneflow.global_function returns "
        + "RemoteBlob or list/tuple/dict nested RemoteBlob only"
    )
