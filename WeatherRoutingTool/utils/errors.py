"""Error codes and exceptions of the Weather Routing Tool (WRT).

Error codes are grouped in ranges of ten so that the origin of an error can be read off its code:

==========  =============================================
Range       Origin
==========  =============================================
0           no error
1 - 9       general (configuration, input/output)
10 - 19     isochrone-based algorithms
20 - 29     genetic algorithm
30 - 39     environmental data
40 - 49     fuel consumption models
90 - 99     unexpected internal errors
==========  =============================================

Every exception class is linked to exactly one error code, and the error message is taken from ``ERROR_MESSAGES``
such that the same error always produces the same message.
"""

from enum import IntEnum


class WRTErrorCodes(IntEnum):
    """Error codes of the Weather Routing Tool."""
    NO_ERROR = 0

    # general
    CONFIG_INVALID = 1  # errors that can be solved by adjusting variables in the config files
    INPUT_FAILED = 2
    OUTPUT_FAILED = 3
    API_USAGE = 4  # errors that results from faulty usage of WRT functionality (function calls with wrong
    # arguments etc.

    # isochrone-based algorithms
    ISOBASED_PRUNING_FAILED = 11
    ISOBASED_OUT_OF_ROUTES = 12
    ISOBASED_DESTINATION_NOT_REACHED = 13
    ISOBASED_INTERNAL = 19  # internal errors in the isobased module

    # genetic algorithm
    GENETIC_INITIAL_POPULATION_FAILED = 21
    GENETIC_OPTIMISATION_FAILED = 22
    GENETIC_MCDM_FAILED = 23
    GENETIC_INTERNAL = 29  # internal errors in the genetic module

    # environmental data
    ENVIRONMENTAL_DATA_FAILED = 30  # errors wrt. handling weather and depth data

    # ship model
    FUEL_CONSUMPTION_INVALID = 40  # errors wrt. fuel-consumption modeling

    # unexpected errors
    INTERNAL = 99  # internal errors (bugs, unexcpected behaviour, ...)


ERROR_MESSAGES = {
    WRTErrorCodes.NO_ERROR: "Weather Routing Tool terminated successfully. {detail}",

    WRTErrorCodes.CONFIG_INVALID: "The configuration of the Weather Routing Tool was invalid. {detail}",
    WRTErrorCodes.INPUT_FAILED: "Input data could not be read. {detail}",
    WRTErrorCodes.OUTPUT_FAILED: "Output data could not be written. {detail}",
    WRTErrorCodes.API_USAGE: "Faulty call to WRT functionality. {detail}",

    WRTErrorCodes.ISOBASED_PRUNING_FAILED: "The pruning of the isochrone-based algorithm failed. {detail}",
    WRTErrorCodes.ISOBASED_OUT_OF_ROUTES: "The isochrone-based algorithm ran out of valid routes. {detail}",
    WRTErrorCodes.ISOBASED_DESTINATION_NOT_REACHED: "The isochrone-based algorithm did not reach the destination. "
                                                    "{detail}",
    WRTErrorCodes.ISOBASED_INTERNAL: "Unexpected internal error of the isochrone-based algorithm. {detail}",

    WRTErrorCodes.GENETIC_INITIAL_POPULATION_FAILED: "No initial population could be generated. {detail}",
    WRTErrorCodes.GENETIC_OPTIMISATION_FAILED: "The optimisation process of the genetic algorithm failed. {detail}",
    WRTErrorCodes.GENETIC_MCDM_FAILED: "The MCDM algorithm failed to find the best-compromise solution. {detail}",
    WRTErrorCodes.GENETIC_INTERNAL: "Unexpected internal error of the genetic algorithm. {detail}",

    WRTErrorCodes.ENVIRONMENTAL_DATA_FAILED: "The environmental data could not be loaded or processed. {detail}",

    WRTErrorCodes.FUEL_CONSUMPTION_INVALID: "Error in the fuel consumption model. {detail}",

    WRTErrorCodes.INTERNAL: "Unexpected internal error of the Weather Routing Tool. {detail}",
}


class WRTError(Exception):
    """Base class of all errors of the Weather Routing Tool.

    Every subclass is linked to exactly one error code via the class attribute ``code``. The error message is taken
    from ``ERROR_MESSAGES`` such that the same error always produces the same message.

    :param detail: additional information that is inserted into the fixed error message
    :type detail: str
    """
    code = WRTErrorCodes.INTERNAL

    def __init__(self, detail: str = ""):
        self.detail = detail
        super().__init__(ERROR_MESSAGES[self.code].format(detail=detail).strip())

    def __str__(self):
        return f"[E{self.code:03d}] {super().__str__()}"


# general
class WRTConfigError(WRTError):
    code = WRTErrorCodes.CONFIG_INVALID

    @classmethod
    def from_validation_error(cls, validation_error):
        """Convert a pydantic ``ValidationError`` into a single ``WRTConfigError``.

        All errors collected by pydantic are listed in the error detail, one line per error, together with the
        field that caused them.

        :param validation_error: error raised by pydantic during the validation of a config
        :type validation_error: pydantic.ValidationError
        :return: config error that lists all validation errors
        :rtype: WRTConfigError
        """
        lines = []
        for err in validation_error.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "<model-level>"
            msg = err["msg"].removeprefix("Value error, ")
            lines.append(f"  {loc}: {msg}")
        detail = f"{validation_error.error_count()} validation error(s):\n" + "\n".join(lines)
        return cls(detail)


class WRTInputError(WRTError):
    code = WRTErrorCodes.INPUT_FAILED


class WRTOutputError(WRTError):
    code = WRTErrorCodes.OUTPUT_FAILED


class WRTInternalError(WRTError):
    code = WRTErrorCodes.INTERNAL


class WRTAPIError(WRTError):
    code = WRTErrorCodes.API_USAGE


# isochrone-based algorithms
class IsobasedRoutingError(WRTError):
    """Base class of errors that terminate the isochrone-based routing regularly.

    For these errors, the routing algorithm still provides the best route at the state at which the error occurred.
    This route is attached to the exception such that it can be written out before the error is reported.

    :param detail: additional information that is inserted into the fixed error message
    :type detail: str
    :param route: best route at the state of the routing algorithm at which the error occurred
    :type route: WeatherRoutingTool.routeparams.RouteParams or None
    """

    def __init__(self, detail: str = "", route=None):
        super().__init__(detail)
        self.route = route


class IsobasedPruningError(IsobasedRoutingError):
    code = WRTErrorCodes.ISOBASED_PRUNING_FAILED


class IsobasedOutOfRoutesError(IsobasedRoutingError):
    code = WRTErrorCodes.ISOBASED_OUT_OF_ROUTES


class IsobasedDestinationNotReachedError(IsobasedRoutingError):
    code = WRTErrorCodes.ISOBASED_DESTINATION_NOT_REACHED


class IsobasedInternalError(WRTInternalError):
    code = WRTErrorCodes.ISOBASED_INTERNAL


# genetic algorithm
class GeneticAlgorithmError(WRTError):
    code = WRTErrorCodes.GENETIC_INTERNAL


class GeneticInitialPopulationError(WRTError):
    code = WRTErrorCodes.GENETIC_INITIAL_POPULATION_FAILED


class GeneticOptimisationError(WRTError):
    code = WRTErrorCodes.GENETIC_OPTIMISATION_FAILED


class GeneticMCDMError(WRTError):
    code = WRTErrorCodes.GENETIC_MCDM_FAILED


class GeneticInternalError(WRTInternalError):
    code = WRTErrorCodes.GENETIC_INTERNAL


# environmental data
class WRTEnvironmentalDataError(WRTError):
    code = WRTErrorCodes.ENVIRONMENTAL_DATA_FAILED


# ship model
class WRTFuelConsumptionError(WRTError):
    code = WRTErrorCodes.FUEL_CONSUMPTION_INVALID
