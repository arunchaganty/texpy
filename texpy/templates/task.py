from typing import List
from texpy.quality_control import QualityControlDecision
from texpy.experiment import TaskHelper


class Task(TaskHelper):
    """
    This entire file defines handles to process a task: when to give
    bonuses, how to parse and aggregate the input and how to check
    quality.
    """
    # region: task specification
    def bonus(input_, response) -> int:
        """
        An (optional) task-specific bonus.
        This could be a bonus for e.g. completing a tutorial or based on the
        length of the task.
        """
        return 0
    # endregion

    # region: data validation and aggregation
    def parse_response(input_: dict, raw_response: dict) -> dict:
        """
        Parses response into a semantically meaningful blob. 
        Henceforth, the value here can be obtained using 
        `response`.

        @param input_: the HIT input
        @param raw_response: A dictionary version of the answer XML returned by AWS.
        Additionally, the response contains the following _Meta field that will 
        always be copied into the result.

        _Meta:
            AssignmentId: str,
            HITId: str,
            WorkerId: str,
            AssignmentStatus: str,
            SubmitTime: str,
            AcceptTime: str,
            WorkerTime: str,
        """
        return response['Answer']

    def aggregate_responses(input_: dict, responses: List[dict]) -> dict:
        """
        Aggregates a list of responses (each of which was parsed by
        parse_response)

        For example, if parse_response returns -> {
            'time': int,
            'value': str,
            },
        aggregate_responses gets as input, input_, {
            'time': List[int],
            'value': List[str]
        }

        See texpy.aggregation for many useful aggregation functions.
        """
        pass

    def metrics(responses):
        """
        Computes metrics on the input.
        Each field of responses matches the schema of parse_response except
        that each field contains a list of lists with individual values.

        For example, if parse_response returns -> {
            'time': int,
            'value': str,
            },
        metrics gets as input, {
            'time': Dict[HITId][Dict[WorkerId, int]],
            'value': Dict[HITId][Dict[WorkerId, str]],
        }

        See texpy.metrics for lots of useful metrics.
        """
        pass
    # endregion

    # region: quality control
    def check_quality(input_: dict, response: dict, agg: dict, metrics: dict) -> List[QualityControlDecision]:
        """
        The quality checking routine.
        @input_     - input object given to the HIT
        @response   - worker's response (as returned by
                      (parse_response())
        @agg        - aggregated response across all workers for this
                      HIT (as returned by aggregate_responses())
        @metrics    - aggregated metrics computed over all the data (as
                      returned by metrics())

        @return     - a list of quality control decisions. Each decision
                      is weighs on accepting or rejecting the task and
                      gives some feedback back to the worker. We reject
                      if any of the returned decisions are to reject the
                      HIT. 
        """
        ret = []
        minimum_expected_time = metrics["ActualTime:mean"] - 2 * metrics["ActualTime:std"]
        if response["ActualTime"] <= minimum_expected_time:
            ret.append(QualityControlDecision(
                    should_approve=false,
                    short_reason="Completed task too quickly",
                    reason=f"We think it requires at least {minimum_expected_time:d} seconds to properly read through the passages, but you took only {response[ActualTime]} seconds to complete the task.",
                    quality_control_update=-50))

        # default case.
        if not ret or all(decision.should_approve for decision in ret):
            ret.append(QualityControlDecision(
                    should_approve=true,
                    short_reason="Approved",
                    reason=f"Task was approved",
                    quality_control_update=+5))
        return ret

    def rejection_email(response, reasons) -> str:
        """
        The text of an email sent out to workers when their work has
        been rejected.

        Note: the maximum length of this list is ? characters.

        If None, do not respond.
        """

        return f"""\
Hello {response[_Meta][WorkerId]},
  We are unfortunately rejecting your work for HIT {response[_Meta][HITId]}.
We've tried our best to ensure that our rejections are fair and
deserved; here are our reasons:
{reasons}
If you still feel treated unfairly, please contact us."""
    # endregion

__all__ = ['Task']
