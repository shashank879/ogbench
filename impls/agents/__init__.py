from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.dhp_1value import DHPv1Agent
from agents.dhp_2value import DHPv2Agent
from agents.dhp_wBuff import DHPBufferAgent
from agents.dhp import DHPAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    dhpv1=DHPv1Agent,
    dhpv2=DHPv2Agent,
    dhpbuff=DHPBufferAgent,
    dhp=DHPAgent,
)
