import sdRDM

from typing import Optional, Union
from typing import List
from typing import Optional
from pydantic import PrivateAttr
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator
from .parameter import Parameter


@forge_signature
class CalibrationModel(sdRDM.DataModel):
    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("calibrationmodelINDEX"),
        xml="@id",
    )

    name: Optional[str] = Field(
        description="Name of the calibration model.", default=None
    )

    equation: Optional[str] = Field(
        description="Equation of the calibration model.", default=None
    )

    parameters: List[Parameter] = Field(
        description="Parameters of the calibration model equation.",
        default_factory=ListPlus,
    )

    __repo__: Optional[str] = PrivateAttr(
        default="git://github.com/FAIRChemistry/CaliPytion.git"
    )

    __commit__: Optional[str] = PrivateAttr(
        default="7c30255582b4c5426d341ff162de2693d5eb3e88"
    )

    def add_to_parameters(
        self,
        name: Optional[str] = None,
        value: Optional[float] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Adds an instance of 'Parameter' to the attribute 'parameters'.

        Args:


            id (str): Unique identifier of the 'Parameter' object. Defaults to 'None'.


            name (Optional[str]): Name of the parameter. Defaults to None


            value (Optional[float]): Value of the parameter. Defaults to None
        """

        params = {"name": name, "value": value}
        if id is not None:
            params["id"] = id
        parameters = [Parameter(**params)]
        self.parameters = self.parameters + parameters
