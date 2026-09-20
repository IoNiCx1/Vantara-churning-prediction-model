from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np 
import pandas as np 

COUNTRIES = [
    "United Kingdom","Germany","France","EIRE","Spain",
    "Netherlands","Belgium","Switzerland","Portugal","Autralia",

]
PRODUCTS = [
    ("85123A","WHITE HANGLING HEART T-LIGHT HOLDER",4.50),
    ("71053","WHITE METAL LANTERN",6.90),
    ("84406B","CREAM CUPID HEART COAT HANGER",3.20),
    ("22423","REGENCY CAKESTAND 3 TIER",14.50),
    ("47566","PARTY BUNKING",8.30),
    ("85099B","JUMBO BAG RED RETROSOPT",2.10),
    ("22720","SET OF 3 CAKE TINS PANTRY DESIGN",9.90),
    ("21730","GLASS STAR FROSTED T-LIGHT HOLDER",5.40),
    ("22197","SMALL POPCORN HOLDER",1.80),
    ("20725","LUNCH BAG RED RETROSOPT",2.50),


]
NON_PORDUCT_CODES = ["POST","D","M","BANK CHARGES","DOT"]
OBS_START = pd.Timestamp("2009-12-01")
OBS_END = pd.Timestamp("2011-12-08")


@dataclass

class Archetype:
    name:str
    mean_gap_days:float
    gap_dispersion:float
    return_propensity:float
    discount_affinity:float
    seasonal_boost_months:tuple
    hazard_per_order:float
    share:float

ARCHETYPE:List[Archetype] = [
    Archetype("loyal_high_value",14,.6,.03,.1,(),.01,.15),
    Archetype("steady_mid_value",28,.8,.05,.2,(),),
    Archetype("occasional_low_value",60,1.2,.04,.3,(),.05,.20),
    Archetype("discounted_hunter",35,1,.06,.8,(),.04,.10),
    Archetype("seasonal_gift_shopper",90,.5,.08,.15,(10,11,12),.02,.10),
    Archetype("at_risk_departing",25,.9,.10,.25,(),.18,.10),
    Archetype("new_customers",35,1,.05,.2,(),.05,.05),

]
def _assign_archetypes(new_customers:int,rng:np.random.Generator)-> List[Archetype]:
    shares = np.array([a.share for a in ARCHETYPE])
    shares = shares/shares.sum()
    idx =rng.choice(len(ARCHETYPES),size = n_customers,p = shares)
    return [ARCHETYPE[i] for i in idx]
