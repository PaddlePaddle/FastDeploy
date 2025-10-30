from fastdeploy.trace.constants import EVENT_TO_STAGE_MAP
from fastdeploy.utils import trace_logger


def print(event, request_id, user):
    """
    记录任务的跟踪日志信息，包括任务名称、开始时间和结束时间等。
    Args:
        task (Task): 待记录的任务对象。
    """
    try:
        trace_logger.info(
            "",
            extra={
                "attributes": {
                    "request_id": f"{request_id}",
                    "user_id": f"{user}",
                    "event": event.value,
                    "stage": EVENT_TO_STAGE_MAP.get(event).value,
                }
            },
        )
    except:
        pass
