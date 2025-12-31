import logging
from typing import List

from happysimulator.events.event import Event
from archive.generate_event import GenerateEvent
from happysimulator.load.arrival_time_provider import ArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.utils.instant import Instant
from happysimulator.utils.ids import get_id

logger = logging.getLogger(__name__)

class Source:
    def __init__(
        self,
        event_provider: EventProvider,
        arrival_time_provider: ArrivalTimeProvider,
        arrival_rate_profile: Profile,
    ):
        self._event_provider = event_provider
        self._profile = arrival_rate_profile
        self._provider: ArrivalTimeProvider = arrival_time_provider
        
    
    # TODO: define a source interface so that it can be used cleanly in the simulatio