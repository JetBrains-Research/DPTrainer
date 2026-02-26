import logging

from opacus.accountants import register_accountant
from riskcal import CTDAccountant

from dptrainer.privacy_arguments import PrivacyArguments

__all__ = ["PrivacyArguments"]

logging.getLogger("opacus.grad_sample.grad_sample_module_fast_gradient_clipping").setLevel(logging.WARNING)

register_accountant("ctd", CTDAccountant)
