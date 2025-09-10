"""
HyperChannels - Discrete-Event HyperChannels for temporal processing.

HyperChannels extend HyperEdges to support discrete-event processing,
enabling temporal dynamics and event-driven communication in the hypergraph.
"""

import asyncio
import numpy as np
from typing import Dict, List, Set, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import deque
import uuid

from .hypergraph import HyperGraph, HyperNode, HyperEdge


class EventType(Enum):
    """Types of events that can flow through HyperChannels."""
    MESSAGE = "message"
    SIGNAL = "signal"  
    UPDATE = "update"
    QUERY = "query"
    RESPONSE = "response"
    LEARNING = "learning"
    SYNCHRONIZATION = "synchronization"
    ALERT = "alert"


@dataclass
class DiscreteEvent:
    """A discrete event that flows through HyperChannels."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    source_node: str
    target_nodes: List[str]
    data: Dict[str, Any]
    priority: int = 0
    ttl: Optional[timedelta] = None
    processed_by: Set[str] = None
    
    def __post_init__(self):
        if self.processed_by is None:
            self.processed_by = set()
    
    def is_expired(self) -> bool:
        """Check if the event has expired."""
        if self.ttl is None:
            return False
        return datetime.now() - self.timestamp > self.ttl
    
    def mark_processed(self, processor_id: str) -> None:
        """Mark the event as processed by a specific processor."""
        self.processed_by.add(processor_id)
    
    def is_processed_by(self, processor_id: str) -> bool:
        """Check if the event was processed by a specific processor."""
        return processor_id in self.processed_by


class EventProcessor:
    """Base class for processing events in HyperChannels."""
    
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.event_handlers: Dict[EventType, Callable] = {}
        self.processing_stats = {
            "events_processed": 0,
            "events_generated": 0,
            "last_activity": None
        }
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register a handler for a specific event type."""
        self.event_handlers[event_type] = handler
    
    async def process_event(self, event: DiscreteEvent) -> List[DiscreteEvent]:
        """Process an event and optionally generate new events."""
        if event.is_expired():
            return []
        
        handler = self.event_handlers.get(event.event_type)
        if handler is None:
            return []
        
        try:
            # Process the event
            result = await handler(event) if asyncio.iscoroutinefunction(handler) else handler(event)
            
            # Mark as processed
            event.mark_processed(self.processor_id)
            
            # Update stats
            self.processing_stats["events_processed"] += 1
            self.processing_stats["last_activity"] = datetime.now()
            
            # Return generated events
            if isinstance(result, list):
                self.processing_stats["events_generated"] += len(result)
                return result
            elif isinstance(result, DiscreteEvent):
                self.processing_stats["events_generated"] += 1
                return [result]
            else:
                return []
        
        except Exception as e:
            # Log error and continue
            print(f"Error processing event {event.event_id}: {e}")
            return []


class HyperChannel(HyperEdge):
    """
    A HyperChannel extends HyperEdge to support discrete-event processing.
    
    HyperChannels can:
    - Queue and process discrete events
    - Maintain temporal ordering
    - Support event filtering and routing
    - Enable synchronized processing across nodes
    """
    
    def __init__(self, channel_id: Optional[str] = None, 
                 nodes: Optional[List[HyperNode]] = None,
                 channel_type: str = "generic",
                 data: Optional[Dict[str, Any]] = None,
                 max_queue_size: int = 1000,
                 processing_mode: str = "async"):  # async, sync, batch
        
        super().__init__(channel_id, nodes, f"channel_{channel_type}", data)
        
        # Channel-specific attributes
        self.channel_type = channel_type
        self.max_queue_size = max_queue_size
        self.processing_mode = processing_mode
        
        # Event queue and processing
        self.event_queue: deque = deque(maxlen=max_queue_size)
        self.event_processors: Dict[str, EventProcessor] = {}
        self.event_filters: List[Callable] = []
        
        # Temporal state
        self.last_event_time: Optional[datetime] = None
        self.processing_active = True
        self.batch_size = 10
        self.batch_timeout = timedelta(seconds=1)
        
        # Statistics
        self.channel_stats = {
            "events_queued": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "avg_processing_time": 0.0,
            "queue_size": 0
        }
        
        # Start processing task
        self._processing_task = None
        if processing_mode in ["async", "batch"]:
            self._start_processing()
    
    def _start_processing(self) -> None:
        """Start the event processing task."""
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_events_loop())
    
    def _stop_processing(self) -> None:
        """Stop the event processing task."""
        self.processing_active = False
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
    
    async def _process_events_loop(self) -> None:
        """Main event processing loop."""
        try:
            while self.processing_active:
                if self.processing_mode == "async":
                    await self._process_next_event()
                elif self.processing_mode == "batch":
                    await self._process_event_batch()
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in event processing loop: {e}")
    
    async def _process_next_event(self) -> None:
        """Process the next event in the queue."""
        if not self.event_queue:
            return
        
        event = self.event_queue.popleft()
        self.channel_stats["queue_size"] = len(self.event_queue)
        
        await self._process_single_event(event)
    
    async def _process_event_batch(self) -> None:
        """Process events in batches."""
        if not self.event_queue:
            await asyncio.sleep(self.batch_timeout.total_seconds())
            return
        
        # Collect batch
        batch = []
        for _ in range(min(self.batch_size, len(self.event_queue))):
            if self.event_queue:
                batch.append(self.event_queue.popleft())
        
        self.channel_stats["queue_size"] = len(self.event_queue)
        
        # Process batch
        tasks = [self._process_single_event(event) for event in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_single_event(self, event: DiscreteEvent) -> None:
        """Process a single event."""
        start_time = datetime.now()
        
        try:
            # Apply filters
            if not self._passes_filters(event):
                return
            
            # Process with all registered processors
            generated_events = []
            for processor in self.event_processors.values():
                new_events = await processor.process_event(event)
                generated_events.extend(new_events)
            
            # Queue generated events
            for new_event in generated_events:
                await self.queue_event(new_event)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_stats(processing_time)
            
            self.channel_stats["events_processed"] += 1
        
        except Exception as e:
            print(f"Error processing event {event.event_id}: {e}")
    
    def _passes_filters(self, event: DiscreteEvent) -> bool:
        """Check if an event passes all filters."""
        for filter_func in self.event_filters:
            try:
                if not filter_func(event):
                    return False
            except Exception:
                continue
        return True
    
    def _update_processing_stats(self, processing_time: float) -> None:
        """Update processing time statistics."""
        current_avg = self.channel_stats["avg_processing_time"]
        processed_count = self.channel_stats["events_processed"]
        
        # Exponential moving average
        alpha = 0.1
        new_avg = alpha * processing_time + (1 - alpha) * current_avg
        self.channel_stats["avg_processing_time"] = new_avg
    
    async def queue_event(self, event: DiscreteEvent) -> bool:
        """Queue an event for processing."""
        if len(self.event_queue) >= self.max_queue_size:
            # Drop oldest event if queue is full
            self.event_queue.popleft()
            self.channel_stats["events_dropped"] += 1
        
        self.event_queue.append(event)
        self.channel_stats["events_queued"] += 1
        self.channel_stats["queue_size"] = len(self.event_queue)
        self.last_event_time = event.timestamp
        
        # Process immediately if in sync mode
        if self.processing_mode == "sync":
            await self._process_single_event(event)
            self.event_queue.pop()  # Remove from queue after processing
            self.channel_stats["queue_size"] = len(self.event_queue)
        
        return True
    
    def create_event(self, event_type: EventType, source_node: str, 
                    target_nodes: List[str], data: Dict[str, Any],
                    priority: int = 0, ttl: Optional[timedelta] = None) -> DiscreteEvent:
        """Create a new discrete event."""
        event = DiscreteEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            source_node=source_node,
            target_nodes=target_nodes,
            data=data,
            priority=priority,
            ttl=ttl
        )
        return event
    
    async def send_event(self, event_type: EventType, source_node: str,
                        target_nodes: Optional[List[str]] = None,
                        data: Optional[Dict[str, Any]] = None,
                        priority: int = 0, ttl: Optional[timedelta] = None) -> str:
        """Send an event through the channel."""
        if target_nodes is None:
            # Send to all nodes in the channel
            target_nodes = [node.id for node in self.get_nodes() if node.id != source_node]
        
        event = self.create_event(
            event_type=event_type,
            source_node=source_node,
            target_nodes=target_nodes,
            data=data or {},
            priority=priority,
            ttl=ttl
        )
        
        await self.queue_event(event)
        return event.event_id
    
    def register_processor(self, processor: EventProcessor) -> None:
        """Register an event processor for this channel."""
        self.event_processors[processor.processor_id] = processor
    
    def unregister_processor(self, processor_id: str) -> None:
        """Unregister an event processor."""
        self.event_processors.pop(processor_id, None)
    
    def add_filter(self, filter_func: Callable[[DiscreteEvent], bool]) -> None:
        """Add an event filter."""
        self.event_filters.append(filter_func)
    
    def remove_filter(self, filter_func: Callable[[DiscreteEvent], bool]) -> None:
        """Remove an event filter."""
        if filter_func in self.event_filters:
            self.event_filters.remove(filter_func)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queue_size": len(self.event_queue),
            "max_queue_size": self.max_queue_size,
            "processing_mode": self.processing_mode,
            "processing_active": self.processing_active,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get channel statistics."""
        return {
            "channel_id": self.id,
            "channel_type": self.channel_type,
            "queue_status": self.get_queue_status(),
            "stats": self.channel_stats.copy(),
            "processors": list(self.event_processors.keys()),
            "filters": len(self.event_filters)
        }
    
    def __del__(self):
        """Cleanup when channel is destroyed."""
        self._stop_processing()


class DiscreteEventChannel(HyperChannel):
    """
    Specialized HyperChannel for discrete-event simulation and processing.
    
    Features:
    - Event scheduling and timing
    - Causal ordering
    - Synchronization primitives
    - Performance monitoring
    """
    
    def __init__(self, channel_id: Optional[str] = None,
                 nodes: Optional[List[HyperNode]] = None,
                 data: Optional[Dict[str, Any]] = None):
        
        super().__init__(
            channel_id=channel_id,
            nodes=nodes,
            channel_type="discrete_event",
            data=data,
            processing_mode="async"
        )
        
        # Discrete-event specific features
        self.simulation_time = 0.0
        self.event_schedule: List[Tuple[float, DiscreteEvent]] = []
        self.causal_dependencies: Dict[str, Set[str]] = {}
        self.synchronization_barriers: Dict[str, Set[str]] = {}
        
        # Performance tracking
        self.latency_measurements: deque = deque(maxlen=1000)
        self.throughput_window = timedelta(seconds=10)
        self.throughput_events: deque = deque()
        
        # Register default event handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default event handlers for common operations."""
        default_processor = EventProcessor("default_channel_processor")
        
        # Message passing handler
        async def handle_message(event: DiscreteEvent) -> List[DiscreteEvent]:
            # Forward message to target nodes
            forwarded_events = []
            for target_id in event.target_nodes:
                target_node = next((n for n in self.get_nodes() if n.id == target_id), None)
                if target_node and hasattr(target_node, 'receive_message'):
                    # Create forwarded event
                    forward_event = self.create_event(
                        event_type=EventType.MESSAGE,
                        source_node=event.source_node,
                        target_nodes=[target_id],
                        data=event.data
                    )
                    forwarded_events.append(forward_event)
            return forwarded_events
        
        # Signal propagation handler
        async def handle_signal(event: DiscreteEvent) -> List[DiscreteEvent]:
            # Propagate signal with decay
            propagated_events = []
            signal_strength = event.data.get("strength", 1.0)
            decay_factor = event.data.get("decay", 0.9)
            
            if signal_strength > 0.1:  # Only propagate strong signals
                for target_id in event.target_nodes:
                    prop_event = self.create_event(
                        event_type=EventType.SIGNAL,
                        source_node=event.source_node,
                        target_nodes=[target_id],
                        data={
                            **event.data,
                            "strength": signal_strength * decay_factor
                        }
                    )
                    propagated_events.append(prop_event)
            return propagated_events
        
        # Learning event handler
        async def handle_learning(event: DiscreteEvent) -> List[DiscreteEvent]:
            # Distribute learning updates
            learning_events = []
            for target_id in event.target_nodes:
                target_node = next((n for n in self.get_nodes() if n.id == target_id), None)
                if target_node and hasattr(target_node, 'learn_from_interaction'):
                    # Trigger learning on target node
                    await target_node.learn_from_interaction(event.data)
            return learning_events
        
        default_processor.register_handler(EventType.MESSAGE, handle_message)
        default_processor.register_handler(EventType.SIGNAL, handle_signal)
        default_processor.register_handler(EventType.LEARNING, handle_learning)
        
        self.register_processor(default_processor)
    
    def schedule_event(self, delay: float, event: DiscreteEvent) -> None:
        """Schedule an event to be processed at a future simulation time."""
        scheduled_time = self.simulation_time + delay
        self.event_schedule.append((scheduled_time, event))
        self.event_schedule.sort(key=lambda x: x[0])  # Keep sorted by time
    
    async def advance_simulation_time(self, target_time: float) -> List[DiscreteEvent]:
        """Advance simulation time and process scheduled events."""
        processed_events = []
        
        while (self.event_schedule and 
               self.event_schedule[0][0] <= target_time):
            
            scheduled_time, event = self.event_schedule.pop(0)
            self.simulation_time = scheduled_time
            
            await self.queue_event(event)
            processed_events.append(event)
        
        self.simulation_time = target_time
        return processed_events
    
    def add_causal_dependency(self, event_id: str, depends_on: str) -> None:
        """Add a causal dependency between events."""
        if event_id not in self.causal_dependencies:
            self.causal_dependencies[event_id] = set()
        self.causal_dependencies[event_id].add(depends_on)
    
    def create_synchronization_barrier(self, barrier_id: str, 
                                     participating_nodes: List[str]) -> None:
        """Create a synchronization barrier for coordinated processing."""
        self.synchronization_barriers[barrier_id] = set(participating_nodes)
    
    async def wait_for_barrier(self, barrier_id: str, node_id: str) -> bool:
        """Wait for a synchronization barrier to be satisfied."""
        if barrier_id not in self.synchronization_barriers:
            return False
        
        barrier_nodes = self.synchronization_barriers[barrier_id]
        barrier_nodes.discard(node_id)  # Remove this node from waiting list
        
        if not barrier_nodes:  # All nodes have reached the barrier
            del self.synchronization_barriers[barrier_id]
            return True
        
        return False
    
    def measure_latency(self, event: DiscreteEvent) -> None:
        """Measure and record event processing latency."""
        latency = (datetime.now() - event.timestamp).total_seconds()
        self.latency_measurements.append(latency)
    
    def update_throughput(self) -> None:
        """Update throughput measurements."""
        now = datetime.now()
        self.throughput_events.append(now)
        
        # Remove old events outside the window
        cutoff = now - self.throughput_window
        while (self.throughput_events and 
               self.throughput_events[0] < cutoff):
            self.throughput_events.popleft()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the channel."""
        metrics = {
            "simulation_time": self.simulation_time,
            "scheduled_events": len(self.event_schedule),
            "causal_dependencies": len(self.causal_dependencies),
            "synchronization_barriers": len(self.synchronization_barriers)
        }
        
        if self.latency_measurements:
            latencies = list(self.latency_measurements)
            metrics["latency"] = {
                "avg": np.mean(latencies),
                "min": np.min(latencies),
                "max": np.max(latencies),
                "std": np.std(latencies)
            }
        
        if self.throughput_events:
            events_per_second = len(self.throughput_events) / self.throughput_window.total_seconds()
            metrics["throughput"] = {
                "events_per_second": events_per_second,
                "window_size": self.throughput_window.total_seconds()
            }
        
        return metrics
    
    def get_extended_statistics(self) -> Dict[str, Any]:
        """Get extended statistics including performance metrics."""
        base_stats = self.get_statistics()
        performance_metrics = self.get_performance_metrics()
        
        return {
            **base_stats,
            "performance": performance_metrics,
            "discrete_event_features": {
                "simulation_enabled": True,
                "causal_ordering": len(self.causal_dependencies) > 0,
                "synchronization_support": len(self.synchronization_barriers) > 0
            }
        }