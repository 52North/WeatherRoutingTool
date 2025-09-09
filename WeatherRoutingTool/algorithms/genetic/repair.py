from pymoo.core.repair import Repair, NoRepair

import logging

logger = logging.getLogger("WRT.genetic.repair")


class RepairFactory:
    @staticmethod
    def get_repair():
        return NoRepair()
