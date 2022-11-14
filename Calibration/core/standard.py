import sdRDM

from typing import Optional, Union
from pydantic import PrivateAttr
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator

from pydantic import Field
from typing import List
from typing import Optional

from .concentrationunits import ConcentrationUnits
from .series import Series


@forge_signature
class Standard(sdRDM.DataModel):
    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("standardINDEX"),
        xml="@id",
    )
    wavelength: Optional[float] = Field(
        description="Detection wavelength.",
        default=None,
    )

    concentration: List[float] = Field(
        description="Concentration of the reactant.",
        default_factory=ListPlus,
    )

    concentration_unit: Optional[ConcentrationUnits] = Field(
        description="Concentration unit.",
        default=None,
    )

    absorption: List[Series] = Field(
        description=(
            "Measured absorption, corresponding to the applied concentration of the"
            " reactant."
        ),
        default_factory=ListPlus,
    )

    __repo__: Optional[str] = PrivateAttr(
        default="git://github.com/FAIRChemistry/datamodel_calibration.git"
    )
    __commit__: Optional[str] = PrivateAttr(
        default="bdadd49e3bc92f57bb84511721888175000cd011"
    )

    def add_to_absorption(self, values: List[float], id: Optional[str] = None) -> None:
        """
        Adds an instance of 'Series' to the attribute 'absorption'.

        Args:
            id (str): Unique identifier of the 'Series' object. Defaults to 'None'.
            values (List[float]): Series representing an array of value.
        """

        params = {
            "values": values,
        }

        if id is not None:
            params["id"] = id

        absorption = [Series(**params)]

        self.absorption = self.absorption + absorption
