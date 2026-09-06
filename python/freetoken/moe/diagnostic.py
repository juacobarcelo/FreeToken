"""Private observation boundary for an exclusive ISP diagnostic worker.

The ordinary runtime leaves observer as None. No tensor copy, event, or timer
is created by this module. Mature expert kernels are unchanged.
"""

observer = None
