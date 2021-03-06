# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import inspect
import socket
import sys
import traceback
from os import environ, remove
from tempfile import mktemp

from .kubejob import KubejobRuntime
from ..model import RunObject
from ..utils import logger
from ..execution import MLClientCtx
from .base import BaseRuntime
from .utils import log_std
from sys import executable
from subprocess import run, PIPE

import importlib.util as imputil
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from nuclio_sdk import Event


class HandlerRuntime(BaseRuntime):
    kind = 'handler'

    def _run(self, runobj: RunObject, execution):
        handler = runobj.spec.handler
        self._force_handler(handler)
        tmp = mktemp('.json')
        environ['MLRUN_META_TMPFILE'] = tmp
        context = MLClientCtx.from_dict(runobj.to_dict(),
                                        rundb=self.spec.rundb,
                                        autocommit=False,
                                        tmp=tmp,
                                        host=socket.gethostname())
        sys.modules[__name__].mlrun_context = context
        sout, serr = exec_from_params(handler, runobj, context)
        log_std(self._db_conn, runobj, sout, serr)
        return context.to_dict()


class LocalRuntime(BaseRuntime):
    kind = 'local'
    _is_remote = False

    def to_job(self, image=''):
        struct = self.to_dict()
        obj = KubejobRuntime.from_dict(struct)
        if image:
            obj.spec.image = image
        return obj

    @property
    def is_deployed(self):
        return True

    def _run(self, runobj: RunObject, execution):
        environ['MLRUN_EXEC_CONFIG'] = runobj.to_json()
        tmp = mktemp('.json')
        environ['MLRUN_META_TMPFILE'] = tmp
        if self.spec.rundb:
            environ['MLRUN_DBPATH'] = self.spec.rundb

        handler = runobj.spec.handler
        if handler:
            mod, fn = load_module(self.spec.command, handler)
            context = MLClientCtx.from_dict(runobj.to_dict(),
                                            rundb=self.spec.rundb,
                                            autocommit=False,
                                            tmp=tmp,
                                            host=socket.gethostname())
            mod.mlrun_context = context
            sout, serr = exec_from_params(fn, runobj, context)
            log_std(self._db_conn, runobj, sout, serr, skip=self.is_child)
            return context.to_dict()

        else:
            if self.spec.mode == 'pass':
                cmd = [self.spec.command]
            else:
                cmd = [executable, self.spec.command]
            sout, serr = run_exec(cmd, self.spec.args)
            log_std(self._db_conn, runobj, sout, serr, skip=self.is_child)

            try:
                with open(tmp) as fp:
                    resp = fp.read()
                remove(tmp)
                if resp:
                    return json.loads(resp)
                logger.error('empty context tmp file')
            except FileNotFoundError:
                logger.info('no context file found')
            return runobj.to_dict()


def load_module(file_name, handler):
    """Load module from file name"""
    path = Path(file_name)
    mod_name = path.name
    if path.suffix:
        mod_name = mod_name[:-len(path.suffix)]
    spec = imputil.spec_from_file_location(mod_name, file_name)
    if spec is None:
        raise ImportError(f'cannot import from {file_name!r}')
    mod = imputil.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, handler)  # Will raise if name not found
    return mod, fn


def run_exec(cmd, args, env=None):
    if args:
        cmd += args
    out = run(cmd, stdout=PIPE, stderr=PIPE, env=env)

    err = out.stderr.decode('utf-8') if out.returncode != 0 else ''
    return out.stdout.decode('utf-8'), err


def run_func(file_name, name='main', args=None, kw=None, *, ctx=None):
    """Run a function from file with args and kw.

    ctx values are injected to module during function run time.
    """
    mod = load_module(file_name)
    fn = getattr(mod, name)  # Will raise if name not found

    if ctx is not None:
        for attr, value in ctx.items():
            setattr(mod, attr, value)

    args = [] if args is None else args
    kw = {} if kw is None else kw

    stdout = StringIO()
    err = ''
    val = None
    with redirect_stdout(stdout):
        try:
            val = fn(*args, **kw)
        except Exception as e:
            logger.error(traceback.format_exc())
            err = str(e)

    return val, stdout.getvalue(), err


def exec_from_params(handler, runobj: RunObject, context: MLClientCtx):
    args_list = get_func_arg(handler, runobj, context)

    stdout = StringIO()
    err = ''
    val = None
    with redirect_stdout(stdout):
        context.set_logger_stream(stdout)
        try:
            val = handler(*args_list)
            context.set_state('completed', commit=False)
        except Exception as e:
            err = str(e)
            logger.error(traceback.format_exc())
            context.set_state(error=err, commit=False)

    context.set_logger_stream(sys.stdout)
    if val:
        context.log_result('return', val)
    context.commit()
    return stdout.getvalue(), err


def get_func_arg(handler, runobj: RunObject, context: MLClientCtx):
    params = runobj.spec.parameters or {}
    inputs = runobj.spec.inputs or {}
    args_list = []
    i = 0
    args = inspect.signature(handler).parameters
    if len(args) > 0 and list(args.keys())[0] == 'context':
        args_list.append(context)
        i += 1
    if len(args) > i + 1 and list(args.keys())[i] == 'event':
        event = Event(runobj.to_dict())
        args_list.append(event)
        i += 1

    for key in list(args.keys())[i:]:
        if args[key].name in params:
            args_list.append(params[key])
        elif args[key].name in inputs:
            obj = context.get_input(key, inputs[key])
            if type(args[key].default) is str:
                filepath = obj.url
                if obj.kind != 'file':
                    dot = filepath.rfind('.')
                    filepath = mktemp() if dot == -1 else \
                        mktemp(filepath[dot:])
                    logger.info('downloading {} to local tmp'.format(obj.url))
                    obj.download(filepath)
                args_list.append(filepath)
            else:
                args_list.append(context.get_input(key, inputs[key]))
        elif args[key].default is not inspect.Parameter.empty:
            args_list.append(args[key].default)
        else:
            args_list.append(None)

    return args_list
