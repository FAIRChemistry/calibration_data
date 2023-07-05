import sdRDM

from typing import List, Optional
from pydantic import Field, PrivateAttr
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


from .concentrationunit import ConcentrationUnit
from .series import Series


@forge_signature
class Spectrum(sdRDM.DataModel):

    """"""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("spectrumINDEX"),
        xml="@id",
    )

    concentration: Optional[float] = Field(
        default=None,
        description="Concentration of the reactant at which the spectrum was recorded",
    )

    wavelength: List[float] = Field(
        default_factory=ListPlus,
        multiple=True,
        description="Wavelengths used for detection",
    )

    concentration_unit: Optional[ConcentrationUnit] = Field(
        default=None,
        description="Concentration unit",
    )

    absorption: List[Series] = Field(
        default_factory=ListPlus,
        multiple=True,
        description="Measured absorption, corresponding to detection wavelengths",
    )

    __repo__: Optional[str] = PrivateAttr(
        default="https://github.com/FAIRChemistry/CaliPytion.git"
    )
    __commit__: Optional[str] = PrivateAttr(
        default="ac2969a829fcebd4fea6bd449c02c1d4fefba4be"
    )

    def add_to_absorption(
        self, values: List[float] = ListPlus(), id: Optional[str] = None
    ) -> None:
        """
        This method adds an object of type 'Series' to attribute absorption

        Args:
            id (str): Unique identifier of the 'Series' object. Defaults to 'None'.
            values (): Series representing an array of value. Defaults to ListPlus()
        """

        params = {
            "values": values,
        }

        if id is not None:
            params["id"] = id

        self.absorption.append(Series(**params))

        return self.absorption[-1]
