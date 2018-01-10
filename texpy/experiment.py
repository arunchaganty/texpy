"""
Handles the representation and definition of experiments.
"""

import os
import yaml
import logging
from importlib.util import spec_from_file_location, module_from_spec
from typing import TextIO, List, cast, Union
from argparse import ArgumentError

from .util import load_jsonl, save_jsonl
from .quality_control import QualityControlDecision

logger = logging.getLogger(__name__)


class TaskHelper:
    """
    This entire file defines handles to process a task: when to give
    bonuses, how to parse and aggregate the input and how to check
    quality.
    """

    # region: task specification
    def bonus(self, input_, response) -> int:
        """
        An (optional) task-specific bonus.
        This could be a bonus for e.g. completing a tutorial or based on the
        length of the task.
        """
        return 0
    # endregion

    # region: data validation and aggregation
    def aggregate_responses(self, input_: dict, raw_responses: List[dict]) -> dict:
        """
        Aggregates multiple raw responses.
        See texpy.aggregation for many useful aggregation functions.

        :param input_: The raw input provided to this task.
        :param raw_responses: A list of raw_responses. Each raw_response is a dictionary with key-value pairs
                              from the form's fields.
        :returns A dictionary that aggregates the responses with the same keys.
        """
        pass

    def make_output(self, input_: dict, agg: dict) -> List[dict]:
        """
        Create well-structured output from input and the aggregated response.

        :param input_:  input given to the HIT
        :param agg   :  return value of aggregate_responses

        :returns     :  A list of objects to be saved in the output data file
        """
        pass

    def compute_metrics(self, inputs: List[dict], outputs: List[List[dict]], agg: List[dict]) -> dict:
        """
        Computes metrics on the input.

        :param inputs   : A list of inputs for the task.
        :param outputs  : A list of outputs; each element contains a list of responses from workers.
        :param agg      : The aggregated output.

        See texpy.metrics for lots of useful metrics.

        :returns a dictionary containing all the metrics we care about for this task.
        """
        pass
    # endregion

    # region: quality control
    def check_quality(self, input_: dict, response: dict, agg: dict, metrics: dict) -> List[QualityControlDecision]:
        """
        The quality checking routine.
        :param input_  : input object given to the HIT
        :param response: A worker's response (an element of outputs)
        :param agg     : Aggregated response for this task (as returned by aggregate_responses())
        :param metrics : Metrics computed over all the data (as returned by compute_metrics())

        :returns       : a list of quality control decisions. Each decision
                         weighs on accepting or rejecting the task and
                         gives some feedback back to the worker. We reject
                         if any of the returned decisions are to reject the
                         HIT.
        """
        pass

    @property
    def rejection_email_format(self) -> str:
        """
        The text of an email sent out to workers when their work has
        been rejected.

        Note: the maximum length of this list is ? characters.

        If None, do not respond.
        """
        return """\
Hello {response[_Meta][WorkerId]},
  We are unfortunately rejecting your work for HIT {response[_Meta][HITId]}.
We've tried our best to ensure that our rejections are fair and
deserved; here are our reasons:
{reasons}
If you still feel treated unfairly, please contact us."""

    def rejection_email(self, response: dict, char_limit: int = 1024) -> str:
        """
        The text of an email sent out to workers when their work has
        been rejected.

        Note: the maximum length of this list is ? characters.

        If None, do not respond.
        """
        template = self.rejection_email_format
        reasons: List[QualityControlDecision] = response["_Meta"].get("QualityControlDecisions", [])

        ret = template.format(
            response=response,
            reasons="\n".join("- {}".format(reason.reason) for reason in reasons)
        )

        if len(ret) >= char_limit:
            logger.warning(f"Could not report full feedback for {response['_Meta']['AssignmentId']};"
                           f" using parsimonious version.")
            ret = template.format(
                response=response,
                reasons="\n".join("- {}".format(reason.short_reason) for reason in reasons)
            )

        if len(ret) >= char_limit:
            logger.warning("Could not report parsimonious feedback for {response['_Meta']['AssignmentId']};"
                           " using truncated version.")

            backup_version = "... (message truncated): if you would like further explanation " \
                f"about our decision, please contact us."
            ret = ret[:(char_limit - len(backup_version))] + backup_version

        assert len(ret) < char_limit

        return ret
    # endregion


class Experiment(object):
    """
    A tex.py experiment is a folder with the following objects:
    # Task specification
        - task.yaml         : specifies the AMT task paraemeters (title,
                              reward, etc.)
        - task.py           : specifies how to parse the task input and
                              output.
        - inputs.jsonl      : the input to run the experiment with.
        - static/           : static resources used to render the task.
            index.html      : the HTML that is rendered as an AMT task.
            js/*            : other miscellaneous JavaScript files --
                              these must be uploaded to some publicly
                              accessible URL to work.
            

    # Generated files
        - hits.jsonl        : metadata about HIT ids and the status of
                              worker assignments. Each line corresponds
                              to a single HIT which in turn corresponds
                              to a single task in inputs.jsonl.
        - outputs.jsonl     : raw output from workers. Again, each line
                              corresponds to a single task specified by
                              a line in inputs.jsonl. If a `task.py` is
                              present, the output here corresponds to
                              the output generated by task.py.
        - data.jsonl        : aggregated output generated from
                              outputs.jsonl.
        - metrics.yaml      : metrics computed on the data as specified
                              in task.py.
    """

    def __init__(self, root: str, type_: str, idx: int):
        """
        Creates a new experiment rooted at @root of type @type_ and index @idx
        """
        self.root = root
        self.type = type_
        self.idx = idx

        self._config = None
        self._helper = None
        self._inputs = None

    def __repr__(self):
        return "<Exp: {}>".format(self.path())

    # region: io
    @property
    def mypath(self) -> str:
        """
        Gets an filename as a property (helpful in format strings)
        """
        return self.path()

    def exists(self, path=None) -> bool:
        """
        Test if a path exists.
        """
        return os.path.exists(self.path(path))

    def ensure_exists(self, path=None):
        """
        Ensure directories leading up to a path exist.
        """
        path = self.path(path)
        if not os.path.exists(path):
            os.makedirs(path)

    def relpath(self, fname=None) -> str:
        """
        Gets path relative to experiment root.
        """
        base_path = os.path.join(self.type, str(self.idx))
        if fname:
            return os.path.join(base_path, fname)
        else:
            return base_path

    def path(self, fname=None) -> str:
        """
        Gets an filename from exp directory rooted at @self.root
        """
        return os.path.join(self.root, self.relpath(fname))

    def open(self, fname: str, *args, **kwargs) -> TextIO:
        """
        Opens a file relative to the experiment directory
        """
        return open(self.path(fname), *args, **kwargs)

    def load(self, fname: str) -> dict:
        with self.open(fname) as f:
            return yaml.safe_load(f)

    def loadl(self, fname: str) -> List[dict]:
        with self.open(fname) as f:
            return load_jsonl(f)

    def store(self, fname: str, obj: dict):
        with self.open(fname, "w") as f:
            yaml.safe_dump(obj, f, indent=2, sort_keys=True)

    def storel(self, fname: str, objs: List[Union[list, dict]]):
        with self.open(fname, "w") as f:
            save_jsonl(f, objs)

    # endregion

    # region: commands
    @classmethod
    def create(cls, root: str, type_: str) -> 'Experiment':
        """
        Create a new experiment of type type_ in root.
        """
        priors = cls.of_type(root, type_)
        if priors:
            exp = cls(root, type_, priors[-1].idx + 1)
        else:
            exp = cls(root, type_, 0)
        exp.ensure_exists()
        return exp

    @classmethod
    def from_path(cls, path, root=None) -> 'Experiment':
        """
        Create an experiment for a file path.
        """
        if root is None:
            root = os.path.dirname(os.path.dirname(path))
            # + 1 for the '/'
            path = path[len(root) + 1:]

        type_ = os.path.dirname(path)
        try:
            idx = int(os.path.basename(path))
        except ValueError:
            raise ValueError(f"{path} is not a valid experiment directory")
        return Experiment(root, type_, int(idx))

    @classmethod
    def of_type(cls, root, type_) -> List['Experiment']:
        """
        Get all experiments of certain type.
        """
        if not os.path.exists(os.path.join(root, type_)): return []

        ret = []
        for dirname in os.listdir(os.path.join(root, type_)):
            try:
                exp = cls.from_path(os.path.join(root, type_, dirname))
                if exp.type == type_:
                    ret.append(exp)
            except ValueError:
                pass
        return sorted(ret, key=lambda e: e.idx)

    # endregion

    # region: Getters for common experiment things.
    @property
    def config(self) -> dict:
        """
        Get the task's configuration
        """
        if self._config is None:
            with self.open("task.yaml") as f:
                self._config = yaml.safe_load(f)
        return self._config

    @property
    def helper(self) -> TaskHelper:
        """
        Loads the task helper
        """
        if self._helper is None:
            # This sequence of magic allows us to import a python module.
            spec = spec_from_file_location("texpy.task", self.path("task.py"))
            assert spec is not None and spec.loader is not None
            module = module_from_spec(spec)
            assert module is not None
            spec.loader.exec_module(module)  # type: ignore

            assert hasattr(module, 'Task')
            self._helper = cast(TaskHelper, module.Task())  # type: ignore
        return self._helper

    @property
    def inputs(self) -> List[dict]:
        """
        Get the task's configuration
        """
        if self._inputs is None:
            self._inputs = self.loadl("inputs.jsonl")
        return self._inputs
    # endregion


def find_experiment(root: str, suffix: str) -> Experiment:
    """
    Finds an experiment of type @type from @root

    Examples:
    SimpleApp -> SimpleApp/<latest>
    SimpleApp/<n>

    Returns an Experiment object
    """
    try:
        exp = Experiment.from_path(os.path.join(root, suffix))
        if not exp.exists():
            raise ArgumentError(None, f"Experiment directory {exp.path()} does not exist")
        logger.info(f"Using experiment at path {exp.path()}")
        return exp
    except ValueError:
        pass

    priors = Experiment.of_type(root, suffix)
    if priors:
        exp = priors[-1]
        logger.info(f"Using experiment at path {exp.path()}")
        return exp
    else:
        raise ArgumentError(None, "No experiments of type {} exist".format(suffix))


__all__ = ["find_experiment", "Experiment", "TaskHelper"]
