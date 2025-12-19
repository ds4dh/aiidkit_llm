
##################################################################
# Code from https://github.com/melhk/AIIDKIT-Multi-states-models #
##################################################################

from enum import Enum, auto
import pandas as pd


class TypeOfTransplantation(Enum):
    LEFT_KIDNEY = auto()
    RIGHT_KIDNEY = auto()
    BOTH_KIDNEYS = auto()
    UNKNOWN = auto()  # Inculding "global consent refused" and "unknown"


class DonorRelatedInfection(Enum):
    YES = auto() # Potential Donor-related Infection Disease (possible/probable/proven according to the definitions)
    NO = auto() # Excluded Donor-related Infection Disease (unlikely, excluded)
    UNKNOWN = auto() # No further analyzed


class RequiredHospitalization(Enum):
    YES = auto()  # Required hospitalization event occurred
    NO = auto()   # No required hospitalization event occurred
    UNKNOWN = auto()  # Required hospitalization event unknown or not specified


class BloodTransfusionEvent(Enum):
    YES = auto()  # Blood transfusion event occurred
    NO = auto()   # No blood transfusion event occurred
    UNKNOWN = auto()  # Blood transfusion event unknown or not specified


class ImmunosuppressionReduced(Enum):
    YES = auto()  # Immunosuppression reduced
    NO = auto()   # Immunosuppression not reduced
    UNKNOWN = auto()  # Immunosuppression reduction status unknown or refused


class Transplantation:
    def __init__(self, patient_ID:int, transplantation_date:pd.Timestamp):
        self._patient_ID = patient_ID # Patient ID associated with this transplantation
        self._transplantation_date = transplantation_date # Date of transplantation 
        self._hospitalization_start_date = pd.NaT
        self._hospitalization_end_date = pd.NaT
        self._type_of_transplantation = None
        self._blood_transfusion_event = None
        self._kidneys = []  # List of kidneys objects associated with this transplantation
    
    @property
    def patient_ID(self) -> int:
        return self._patient_ID

    @patient_ID.setter
    def patient_ID(self, patient_ID:int) -> None:
        self._patient_ID = patient_ID

    @property
    def transplantation_date(self) -> pd.Timestamp:
        return self._transplantation_date
    
    @transplantation_date.setter
    def transplantation_date(self, transplantation_date:pd.Timestamp) -> None:
        self._transplantation_date = transplantation_date

    @property
    def hospitalization_start_date(self) -> pd.Timestamp:
        return self._hospitalization_start_date

    @hospitalization_start_date.setter
    def hospitalization_start_date(self, hospitalization_start_date:pd.Timestamp) -> None:
        self._hospitalization_start_date = hospitalization_start_date

    @property
    def hospitalization_end_date(self) -> pd.Timestamp:
        return self._hospitalization_end_date
    
    @hospitalization_end_date.setter
    def hospitalization_end_date(self, hospitalization_end_date:pd.Timestamp) -> None:
        self._hospitalization_end_date = hospitalization_end_date

    @property
    def type_of_transplantation(self) -> TypeOfTransplantation:
        return self._type_of_transplantation
    
    @type_of_transplantation.setter
    def type_of_transplantation(self, type_of_transplantation:TypeOfTransplantation) -> None:
        self._type_of_transplantation = type_of_transplantation

    @property
    def kidneys(self):
        return self._kidneys
    
    @kidneys.setter
    def kidneys(self, kidneys:list):
        self._kidneys = kidneys

    @property
    def blood_transfusion_event(self) -> BloodTransfusionEvent:
        return self._blood_transfusion_event
    
    @blood_transfusion_event.setter
    def blood_transfusion_event(self, blood_transfusion_event:BloodTransfusionEvent) -> None:
        self._blood_transfusion_event = blood_transfusion_event

    def add_kidney(self, kidney):
        """ Add a kidney to the transplantation """
        if kidney not in self._kidneys:
            self._kidneys.append(kidney)

    def get_hospitalization_duration(self) -> int:
        """ Calculate the duration of hospitalization for this transplantation in days """
        if pd.isna(self._hospitalization_start_date) or pd.isna(self._hospitalization_end_date):
            raise ValueError("Hospitalization start or end date is not set.")
        days = (self._hospitalization_end_date - self._hospitalization_start_date).days
        if days < 1:
            raise ValueError("Hospitalization end date cannot be before start date.")
        return days


class Pathogen():
    """ Base class for all pathogens
    """
    def __init__(self, pathogen_type:str=None) -> None:
        self._pathogen_type = pathogen_type

    @property
    def pathogen_type(self) -> str:
        return self._pathogen_type

    @pathogen_type.setter
    def pathogen_type(self, pathogen_type:str=None) -> None:
        self._pathogen_type = pathogen_type


class Bacteria(Pathogen):
    def __init__(
        self,
        pathogen_type: str=None,
        ESBL_resistance: str=None,
        multidrug_resistance: str=None,
        staph_aureus_resistance_phenotype: str=None,
        enterococcis_resistance_phenotype: str=None,
        cpe_resistance_phenotype: str=None,
    ):
        super().__init__(pathogen_type)
        self._ESBL_resistance = ESBL_resistance 
        self._multidrug_resistance = multidrug_resistance 
        self._staph_aureus_resistance_phenotype = staph_aureus_resistance_phenotype 
        self._enterococcis_resistance_phenotype = enterococcis_resistance_phenotype 
        self._cpe_resistance_phenotype = cpe_resistance_phenotype 
    
    @property
    def ESBL_resistance(self) -> str:
        return self._ESBL_resistance
    
    @property
    def multidrug_resistance(self) -> str:
        return self._multidrug_resistance
    
    @property
    def staph_aureus_resistance_phenotype(self) -> str:
        return self._staph_aureus_resistance_phenotype
    
    @property
    def enterococcis_resistance_phenotype(self) -> str:
        return self._enterococcis_resistance_phenotype
    
    @property
    def cpe_resistance_phenotype(self) -> str:
        return self._cpe_resistance_phenotype


class Virus(Pathogen):
    def __init__(
        self,
        pathogen_type: str,
        viral_primary_infection: bool,
    ):
        super().__init__(pathogen_type)
        self._viral_primary_infection = viral_primary_infection # bool


class Fungus(Pathogen):
    def __init__(
        self,
        pathogen_type: str, 
        antifungal_treatment:str,
    ):
        super().__init__(pathogen_type)
        antifungal_treatment = antifungal_treatment
    

class Parasite(Pathogen):
    def __init__(
        self,
        pathogen_type: str,
        parasitic_primary_infection: bool,
    ):
        super().__init__(pathogen_type)
        self._parasitic_primary_infection = parasitic_primary_infection


class InfectionType(Enum):
    """
    Applicable categories of clinical type infections. 
    Not all categories are applicable to all pathogens.
    """
    ASYMPTOMATIC = auto()
    COLONIZATION = auto()
    PRIMARY_INFECTION = auto()
    PROBABLE_DISEASE = auto() 
    PROVEN_DISEASE = auto() 
    POSSIBLE_DISEASE = auto()
    VIRAL_SYNDROME = auto()
    FEVER_NEUTROPENIA = auto()
    UNKNOWN = auto()  # Unknown or not specified
    NO_INFECTION = auto()  # No infection detected or not applicable


class InfectionSite(Enum):
    EYE = auto()
    LIVER = auto()
    BONE_AND_JOINT = auto()
    URINARY_TRACT = auto()
    BLOODSTREAM = auto()
    MUSCULOCUTANEOUS = auto()
    RESPIRATORY_TRACT = auto() 
    PROSTHETIC_DEVICE = auto()  # Including prosthetic heart valve, joint, etc.
    CATHETER = auto()  # Including central venous catheter, urinary catheter, etc.
    SURGICAL_SITE_INFECTION = auto()
    GASTROINTESTINAL_TRACT = auto()
    HEART = auto()  # Including endocarditis
    CENTRAL_NERVOUS_SYSTEM = auto()  # Including meningitis, encephalitis, etc.
    OTHER = auto() 
    UNIDENTIFIED = auto()  # Unidentified infection site, not specified or unknown


class Infection():
    def __init__(
        self,
        patient_ID: int, 
        infection_date: pd.Timestamp,
        clinical_infection_type: InfectionType,
        required_hospitalization: RequiredHospitalization,
        infection_sites: list[InfectionSite],
        donor_related_infection: DonorRelatedInfection,
        immunosuppression_reduced: ImmunosuppressionReduced
    ):
                
        if not self.validate_clinical_infection_type(clinical_infection_type):
            raise ValueError(
                f"Clinical infection type {clinical_infection_type} "
                f"is not valid for pathogen type {type(self)}"
            )

        if not isinstance(patient_ID, int):
            raise TypeError(f"Patient ID must be an int, got {type(patient_ID)}")
        self._patient_ID = patient_ID
        
        self._infection_date = infection_date  # pd.Timestamp

        if not isinstance(clinical_infection_type, InfectionType):
            raise TypeError(
                "Clinical infection type must be an InfectionType enum, "
                f"got {type(clinical_infection_type)}"
            )
        self._clinical_infection_type = clinical_infection_type  # InfectionType enum

        if not isinstance(required_hospitalization, RequiredHospitalization):
            raise TypeError(
                "Required hospitalization must be an RequieredHosiptalisation enum, "
                f"got {type(required_hospitalization)}"
            )
        self._required_hospitalization = required_hospitalization  # RequiredHospitalization enum

        self._infection_sites = infection_sites  # List of InfectionSite enums, if any 
        self._donor_related_infection = donor_related_infection  # Enum DonorRelatedInfection, if any
        self._immunosuppression_reduced = immunosuppression_reduced

    @property
    def patient_ID(self) -> int:
        return self._patient_ID

    @property
    def infection_date(self) -> pd.Timestamp:
        return self._infection_date
    
    @property
    def clinical_infection_type(self) -> InfectionType:
        return self._clinical_infection_type
    
    @property
    def required_hospitalization(self) -> RequiredHospitalization:
        return self._required_hospitalization
    
    @property
    def donor_related_infection(self) -> DonorRelatedInfection:
        return self._donor_related_infection

    @property
    def immunosuppression_reduced(self) -> ImmunosuppressionReduced:
        return self._immunosuppression_reduced
    
    @property
    def infection_sites(self) -> list[InfectionSite]:
        return self._infection_sites

    def validate_clinical_infection_type(
        self,
        clinical_infection_type: InfectionType,
    ) -> None:
        """
        Validates the clinical infection type against the infection's pathogen type.
        Raises a ValueError if the clinical infection type is not valid for the pathogen type.
        """
        return clinical_infection_type in [
            InfectionType.PROBABLE_DISEASE, InfectionType.UNKNOWN,
            InfectionType.COLONIZATION, InfectionType.FEVER_NEUTROPENIA,
            InfectionType.POSSIBLE_DISEASE,
        ]

    def get_time_since_transplantation(self, transplantation: Transplantation) -> int:
        """
        Returns the time in days since the transplantation.
        If the infection date is before the transplantation date, returns 0.
        """
        if self.infection_date < transplantation.transplantation_date:
            return 0
        return (self.infection_date - transplantation.transplantation_date).days


class ViralInfection(Infection):
    def __init__(
        self, 
        patient_ID: int, 
        infection_date: pd.Timestamp, 
        clinical_infection_type: InfectionType,
        required_hospitalization: RequiredHospitalization,
        infection_sites: list[InfectionSite],
        donor_related_infection: DonorRelatedInfection,
        immunosuppression_reduced: ImmunosuppressionReduced,
        pathogens: list[Virus],
        antiviral_treatment: bool
    ) -> None:
        
        super().__init__(
            patient_ID=patient_ID,
            infection_date=infection_date,
            clinical_infection_type=clinical_infection_type, 
            required_hospitalization=required_hospitalization, 
            infection_sites=infection_sites,
            donor_related_infection=donor_related_infection,
            immunosuppression_reduced=immunosuppression_reduced,
        )

        self._pathogens = pathogens # List of Virus objects
        self._antiviral_treatment = antiviral_treatment
        self._clinically_significant = self.is_clinically_significant()
    
    @property
    def pathogens(self) -> list[Virus]:
        return self._pathogens

    @property
    def clinically_significant(self) -> bool:
        return self._clinically_significant

    def validate_clinical_infection_type(
        self,
        clinical_infection_type,
    ):
        return clinical_infection_type in [
            InfectionType.ASYMPTOMATIC, InfectionType.PRIMARY_INFECTION,
            InfectionType.PROBABLE_DISEASE, InfectionType.PROVEN_DISEASE,
            InfectionType.VIRAL_SYNDROME, InfectionType.UNKNOWN,
        ]
    
    def is_clinically_significant(self) -> bool:
        if self.clinical_infection_type in [
            InfectionType.PROBABLE_DISEASE,
            InfectionType.PROVEN_DISEASE,
            InfectionType.VIRAL_SYNDROME,
        ]:
            return True
        return False
    

class BacterialInfection(Infection):
    def __init__(
        self,
        patient_ID: int, 
        infection_date: pd.Timestamp, 
        clinical_infection_type: InfectionType,
        required_hospitalization: RequiredHospitalization,
        infection_sites: list[InfectionSite],
        donor_related_infection: DonorRelatedInfection,
        immunosuppression_reduced: ImmunosuppressionReduced,
        pathogens: list[Bacteria], 
        antibacterial_treatment: bool,
    ) -> None:
                 
        super().__init__(
            patient_ID=patient_ID,
            infection_date=infection_date,
            clinical_infection_type=clinical_infection_type, 
            required_hospitalization=required_hospitalization,
            infection_sites=infection_sites,
            donor_related_infection=donor_related_infection,
            immunosuppression_reduced=immunosuppression_reduced,
        )
        
        self._pathogens = pathogens # List of Bacteria objects 
        self._antibacterial_treatment = antibacterial_treatment
        self._clinically_significant = self.is_clinically_significant()
    
    @property
    def pathogens(self) -> list[Bacteria]:
        return self._pathogens

    @property
    def clinically_significant(self) -> bool:
        return self._clinically_significant

    def validate_clinical_infection_type(
        self,
        clinical_infection_type: InfectionType,
    ) -> bool:
        return clinical_infection_type in [
            InfectionType.COLONIZATION,
            InfectionType.PROVEN_DISEASE,
            InfectionType.UNKNOWN,
        ]
    
    def is_clinically_significant(self) -> bool:
        if self.clinical_infection_type in [InfectionType.PROVEN_DISEASE]:
            return True
        return False


class FungalInfection(Infection):
    def __init__(
        self,
        patient_ID: int, 
        infection_date: pd.Timestamp,
        clinical_infection_type: InfectionType,
        required_hospitalization: RequiredHospitalization,
        infection_sites: list[InfectionSite],
        donor_related_infection: DonorRelatedInfection,
        immunosuppression_reduced: ImmunosuppressionReduced,
        pathogens: list[Fungus], 
        antifungal_treatment: bool, 
    ) -> None:
        super().__init__(
            patient_ID=patient_ID, 
            infection_date=infection_date, 
            clinical_infection_type=clinical_infection_type, 
            required_hospitalization=required_hospitalization,
            infection_sites=infection_sites,
            donor_related_infection=donor_related_infection,
            immunosuppression_reduced=immunosuppression_reduced,
        )

        self._pathogens = pathogens
        self._antifungal_treatment = antifungal_treatment
        self._clinically_significant = self.is_clinically_significant()

    @property
    def pathogens(self) -> list[Fungus]:
        return self._pathogens

    @property
    def clinically_significant(self) -> bool:
        return self._clinically_significant

    def validate_clinical_infection_type(
        self,
        clinical_infection_type: InfectionType,
    ) -> bool:
        return clinical_infection_type in [
            InfectionType.COLONIZATION, InfectionType.PROBABLE_DISEASE,
            InfectionType.PROVEN_DISEASE, InfectionType.POSSIBLE_DISEASE,
            InfectionType.UNKNOWN,
        ]
    
    def is_clinically_significant(self) -> bool:

        # Special case for colonizations in fungi
        if self.clinical_infection_type in [InfectionType.COLONIZATION]\
        and isinstance(self, FungalInfection):
            critical_pathogen_types = {
                "Pneumocystis sp", "Aspergillus fumigatus",
                "Aspergillus non-fumigatus", "Zygomycetes",
            }
            pathogen_types = {p.pathogen_type for p in self.pathogens}
            if pathogen_types & critical_pathogen_types:  # set intersection
                return True

        if self.clinical_infection_type in [
            InfectionType.PROVEN_DISEASE, InfectionType.POSSIBLE_DISEASE
        ]:
            return True
        
        return False


class ParasiticInfection(Infection):
    def __init__(
        self, 
        patient_ID:int,
        infection_date:pd.Timestamp,
        clinical_infection_type:InfectionType,
        required_hospitalization:RequiredHospitalization,
        infection_sites:list[InfectionSite],
        donor_related_infection:DonorRelatedInfection,
        immunosuppression_reduced:ImmunosuppressionReduced,
        pathogens:list[Parasite], 
        antiparasitic_treatment:bool, 
    ) -> None:
        super().__init__(
            patient_ID=patient_ID,
            infection_date=infection_date,
            clinical_infection_type=clinical_infection_type,
            required_hospitalization=required_hospitalization,
            infection_sites=infection_sites,
            donor_related_infection=donor_related_infection,
            immunosuppression_reduced=immunosuppression_reduced,
        )
        
        self._pathogens = pathogens 
        self._antiparasitic_treatment = antiparasitic_treatment 
        self._clinically_significant = False

    @property
    def pathogens(self) -> list[Parasite]:
        return self._pathogens

    @property
    def clinically_significant(self) -> bool:
        return self._clinically_significant

    def validate_clinical_infection_type(
        self,
        clinical_infection_type: InfectionType,
    ) -> bool:
        return clinical_infection_type in [
            InfectionType.PROBABLE_DISEASE,
            InfectionType.PROVEN_DISEASE,
            InfectionType.UNKNOWN,
        ]
