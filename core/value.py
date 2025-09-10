"""
Core AtomSpace value types for AtomBot.

Values represent mutable data that can be attached to atoms and flow between them.
This enables dynamic state and communication in the agent network.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional
import uuid
import asyncio
from datetime import datetime


class Value(ABC):
    """Abstract base class for all AtomSpace values."""
    
    def __init__(self, value: Any = None):
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self._value = value
        self._subscribers: List[callable] = []
    
    @property
    def value(self) -> Any:
        """Get the current value."""
        return self._value
    
    @value.setter 
    def value(self, new_value: Any) -> None:
        """Set a new value and notify subscribers."""
        old_value = self._value
        self._value = new_value
        self.updated_at = datetime.now()
        self._notify_subscribers(old_value, new_value)
    
    def subscribe(self, callback: callable) -> None:
        """Subscribe to value changes."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: callable) -> None:
        """Unsubscribe from value changes."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def _notify_subscribers(self, old_value: Any, new_value: Any) -> None:
        """Notify all subscribers of value changes."""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(old_value, new_value))
                else:
                    callback(old_value, new_value)
            except Exception as e:
                # Log error but don't let one subscriber break others
                print(f"Error notifying subscriber: {e}")
    
    @abstractmethod
    def __str__(self) -> str:
        """String representation of the value."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"


class FloatValue(Value):
    """A floating-point numeric value."""
    
    def __init__(self, value: float = 0.0):
        super().__init__(float(value))
    
    def __str__(self) -> str:
        return str(self._value)
    
    def __float__(self) -> float:
        return self._value
    
    def __add__(self, other: Union['FloatValue', float]) -> 'FloatValue':
        if isinstance(other, FloatValue):
            return FloatValue(self._value + other._value)
        return FloatValue(self._value + float(other))
    
    def __mul__(self, other: Union['FloatValue', float]) -> 'FloatValue':
        if isinstance(other, FloatValue):
            return FloatValue(self._value * other._value)
        return FloatValue(self._value * float(other))


class StringValue(Value):
    """A string text value."""
    
    def __init__(self, value: str = ""):
        super().__init__(str(value))
    
    def __str__(self) -> str:
        return self._value
    
    def __len__(self) -> int:
        return len(self._value)
    
    def __add__(self, other: Union['StringValue', str]) -> 'StringValue':
        if isinstance(other, StringValue):
            return StringValue(self._value + other._value)
        return StringValue(self._value + str(other))


class LinkValue(Value):
    """A value that references other atoms or values."""
    
    def __init__(self, targets: Optional[List[Any]] = None):
        super().__init__(targets or [])
    
    def __str__(self) -> str:
        return f"LinkValue({len(self._value)} targets)"
    
    def add_target(self, target: Any) -> None:
        """Add a target to this link value."""
        if target not in self._value:
            self._value.append(target)
            self.updated_at = datetime.now()
    
    def remove_target(self, target: Any) -> None:
        """Remove a target from this link value."""
        if target in self._value:
            self._value.remove(target)
            self.updated_at = datetime.now()
    
    def get_targets(self) -> List[Any]:
        """Get all targets in this link."""
        return self._value.copy()


class TruthValue(Value):
    """A truth value with strength and confidence components (from OpenCog tradition)."""
    
    def __init__(self, strength: float = 0.0, confidence: float = 0.0):
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0,1]
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
        super().__init__({"strength": self.strength, "confidence": self.confidence})
    
    def __str__(self) -> str:
        return f"TV({self.strength:.3f}, {self.confidence:.3f})"
    
    def is_true(self) -> bool:
        """Check if this truth value represents 'true'."""
        return self.strength > 0.5 and self.confidence > 0.5
    
    def is_false(self) -> bool:
        """Check if this truth value represents 'false'."""
        return self.strength < 0.5 and self.confidence > 0.5
    
    def is_unknown(self) -> bool:
        """Check if this truth value represents 'unknown'."""
        return self.confidence < 0.5


class StreamValue(Value):
    """A value that represents a stream of data flowing through the network."""
    
    def __init__(self, stream_type: str = "data"):
        self.stream_type = stream_type
        self._buffer: List[Any] = []
        self._max_buffer_size = 1000
        super().__init__(None)
    
    def push(self, data: Any) -> None:
        """Push new data to the stream."""
        self._buffer.append({
            "data": data,
            "timestamp": datetime.now()
        })
        
        # Keep buffer from growing too large
        if len(self._buffer) > self._max_buffer_size:
            self._buffer.pop(0)
        
        self.updated_at = datetime.now()
        self._notify_subscribers(None, data)
    
    def peek(self, count: int = 1) -> List[Any]:
        """Peek at the most recent items without removing them."""
        return [item["data"] for item in self._buffer[-count:]]
    
    def drain(self, count: Optional[int] = None) -> List[Any]:
        """Drain items from the stream buffer."""
        if count is None:
            result = [item["data"] for item in self._buffer]
            self._buffer.clear()
        else:
            result = [item["data"] for item in self._buffer[-count:]]
            self._buffer = self._buffer[:-count] if count < len(self._buffer) else []
        
        return result
    
    def __str__(self) -> str:
        return f"StreamValue({self.stream_type}, {len(self._buffer)} items)"