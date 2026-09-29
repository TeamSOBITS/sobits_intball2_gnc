#pragma once

#include <cstdlib>

namespace sobits_intball2_gnc::common
{

// Diagnostic trace to stderr, on only when MINCO_REBOUND_TRACE is set.
inline bool reboundTraceEnabled()
{
    static const bool enabled = std::getenv("MINCO_REBOUND_TRACE") != nullptr;
    return enabled;
}

}  // namespace sobits_intball2_gnc::common
