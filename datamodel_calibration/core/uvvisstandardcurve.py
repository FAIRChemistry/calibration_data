from typing import Optional
from pydantic import PrivateAttr
from sdRDM.base.listplus import ListPlus
from pydantic import Field
from typing import List
from .series import Series
from .standardcurve import StandardCurve


class UVVisStandardCurve(StandardCurve):

    absorption: List[Series] = Field(
        description="Measured absorption, corresponding to the applied concentration.",
        default_factory=ListPlus,
    )

    __repo__: Optional[str] = PrivateAttr(
        default="git://github.com/FAIRChemistry/datamodel_calibration.git"
    )
    __commit__: Optional[str] = PrivateAttr(
        default="789d5828c81656f3d5269c5cfe1b8b83d5a8ed08"
    )

    def add_to_absorption(
        self,
        values: List[float] = ListPlus(),
    ) -> None:
        """
        Adds an instance of 'Series' to the attribute 'absorption'.

        Args:
            values (List[float]): Series representing an array of values.
        """

        self.absorption.append(
            Series(
                values=values,
            )
        )
