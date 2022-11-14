import sdRDM

from typing import Optional, Union
from typing import List
from typing import Optional
from pydantic import PrivateAttr
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator
from pydantic.types import PositiveFloat
from .device import Device
from .spectrum import Spectrum
from .standard import Standard
from .temperatureunits import TemperatureUnits


@forge_signature
class Calibration(sdRDM.DataModel):
    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("calibrationINDEX"),
        xml="@id",
    )

    reactant_id: Optional[str] = Field(
        description="Unique identifier of the calibrated reactant.", default=None
    )

    date: Optional[str] = Field(
        description="Date when the calibration data was measured", default=None
    )

    pH: Optional[PositiveFloat] = Field(description="pH of solution.", default=None)

    temperature: Optional[PositiveFloat] = Field(
        description="Temperature during calibration.", default=None
    )

    temperature_unit: Optional[TemperatureUnits] = Field(
        description="Temperature unit.", default=None
    )

    device: Optional[Device] = Field(
        description="Device object, containing information about the analytic device.",
        default=None,
    )

    standard: List[Standard] = Field(
        description="Standard data of a substance.", default_factory=ListPlus
    )

    spectrum: Optional[Spectrum] = Field(
        description="Spectrum data of a substance.", default=None
    )

    __repo__: Optional[str] = PrivateAttr(
        default="git://github.com/FAIRChemistry/datamodel_calibration.git"
    )

    __commit__: Optional[str] = PrivateAttr(
        default="bdadd49e3bc92f57bb84511721888175000cd011"
    )
