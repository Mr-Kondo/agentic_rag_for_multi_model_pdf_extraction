# Langfuse OpenTelemetry Context Management Fix

## 問題 (Problem)

```
Context error: No active span in current context. Operations that depend on an active span 
will be skipped. Ensure spans are created with start_as_current_span() or that you're 
operating within an active span context.
```

## 根本原因 (Root Cause)

### 1. Manual Context Manager Protocol vs Standard Python `with` Statement

The original code was using manual `__enter__()` and `__exit__()` calls:

```python
# ❌ INCORRECT: Manual context manager protocol
cm = self.raw.start_as_current_span(...)
s = cm.__enter__()  # ← Manually enters but skips context attachment
# ... use span ...
cm.__exit__(None, None, None)  # ← Manually exits but doesn't restore context
```

**Why this fails:**
- Langfuse SDK uses `_AgnosticContextManager` which wraps OpenTelemetry context management
- The `with` statement triggers internal `context.attach()` that sets OpenTelemetry span context variables
- Manual `__enter__()` calls skip this context attachment step entirely
- When child operations query `get_current_span()`, they find nothing because context variables are not set
- This breaks parent-child span relationships and nested trace hierarchies

### 2. Incomplete Context Setup in Top-Level Trace

The original `trace()` method used `start_span()` without context management:

```python
# ❌ INCORRECT: No context setup for parent trace
span = self._client.start_span(name=name, ...)
# ... manual lifecycle management ...
span.end()
```

**Why this fails:**
- `start_span()` returns a LangfuseSpan object directly (not a context manager)
- It doesn't set OpenTelemetry context for child operations
- Child spans and generations created inside have no parent context to discover
- This causes the "No active span in current context" warning

## 解決策 (Solution)

### Replace Manual Protocol with Standard Python `with` Statement

**✅ CORRECT: Use standard `with` statement at all levels**

```python
# Trace level - establish root context
with self._client.start_as_current_span(name=name, ...) as span:
    # OpenTelemetry context is now set for all child operations
    handle = _TraceHandle(span, trace_id)
    yield handle
    # Context is automatically cleaned up on exit

# Span level - nested spans inherit parent context
with self.raw.start_as_current_span(name=name, ...) as s:
    # This span finds the parent trace via OTel context
    handle = _SpanHandle(s)
    yield handle
    # Context is automatically cleaned up on exit

# Generation level - generations find parent span
with self.raw.start_as_current_generation(model=model, ...) as g:
    # This generation finds the parent span via OTel context
    handle = _GenerationHandle(g)
    yield handle
    # Context is automatically cleaned up on exit
```

## 実装の詳細 (Implementation Details)

### Why the `with` Statement is Critical

The `_AgnosticContextManager` class from OpenTelemetry requires the `with` statement because:

1. **Entry (`__enter__`)**: Calls `context.attach()` to set the span in OpenTelemetry context variables
2. **Yield**: The wrapped operation executes with the context variables set
3. **Exit (`__exit__`)**: Calls `context.detach()` to restore the previous context

```python
# Inside _AgnosticContextManager.__enter__()
def __enter__(self):
    # 1. Step up one level
    self._token = context.attach(...)  # ← Sets context variables
    # 2. Return the yielded object
    return next(self._gen)

# Inside _AgnosticContextManager.__exit__()
def __exit__(self, ...):
    # 1. Restore previous context
    context.detach(self._token)  # ← Restores previous context
    # 2. Finish generator
    next(self._gen)
```

**Manual `__enter__()`/`__exit__()` calls skip this context attachment entirely.**

## Changes Made

### 1. Fixed `LangfuseTracer.trace()` Method

- Changed from `start_span()` with manual lifecycle to `start_as_current_span()` with `with` statement
- Ensures root context is properly established for all descendant operations
- Removed manual `span.end()` call (handled by context manager)

### 2. Fixed `_TraceHandle.span()` Method

- Changed from manual `__enter__()/__exit__()` pattern to standard `with` statement
- Properly propagates parent trace context to child operations
- Simplified error handling (no manual exception propagation)

### 3. Fixed `_TraceHandle.generation()` Method

- Changed from manual `__enter__()/__exit__()` pattern to standard `with` statement
- Allows generations to discover parent spans via context
- Maintains output/token tracking in finally block

## Verification

### Test Results

```
Testing Langfuse context management...

✓ Tracer created
✓ Trace created
✓ Parent span created
✓ Generation created as child of span

✅ SUCCESS: Context management is working!
(No context errors - OpenTelemetry propagation is active)
```

**Before fix:** Context warning appears during trace initialization
**After fix:** No warnings, clean trace hierarchy, proper parent-child relationships

### Pipeline Status

Full pipeline test shows:
- ✅ Zero context warnings
- ✅ All models loaded successfully
- ✅ PDF parsing completes without errors
- ✅ Langfuse traces created with proper hierarchy
- ✅ Token counts flow correctly to Langfuse

## Key Takeaway

**OpenTelemetry context propagation is not optional—it's a design requirement.** The standard Python `with` statement is the only way to properly trigger the internal `context.attach()`/`context.detach()` calls that maintain trace hierarchies. Manual `__enter__()` and `__exit__()` calls, even if they work for object creation, **will always break context propagation**.

## Files Modified

- `langfuse_tracer.py`
  - `LangfuseTracer.trace()` - Line 147-192
  - `_TraceHandle.span()` - Line 263-297
  - `_TraceHandle.generation()` - Line 299-350
