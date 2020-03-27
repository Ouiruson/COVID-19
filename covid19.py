#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from importlib import reload
#===============================================================================
homedir, datadir = "/home/rwilson/", "/data/rwilson/"   # ciclad
homedir, datadir = "/home/riw/",  "/home/riw/"          # local
sys.path.insert(0,homedir + "program/python/")     # répertoire des libs
sys.path.insert(0,homedir + "in-situ/python/")     # répertoire des common
#sys.path.insert(0,"/home/riw/in-situ/python/")    # répertoire des libs
import modules.signal_lib as sig
import modules.io_lib     as iol
import modules.plot_lib   as pll
import modules.io_lib     as pll
reload(iol); reload(sig)

def population_pays(pays):
   """ retourne la population d'un pays """
   pop = {
     "China"       : 1401899480,
     "India"       : 1360174677,
     "US"          : 329505107,
     "Indonesia"   : 266911900,
     "Pakistan"    : 219049520,
     "Brasil"      : 211296996,
     "Nigeria"     : 206139587,
     "Bengladesh"  : 168321286,
     "Russia"      : 146745098,
     "Mexico"      : 126577691,
     "Japan"       : 125950000,
     "Iran"        : 83306510,
     "Turkey"      : 83154997,
     "Germany"     : 83149300,
     "France"      : 67069000,
     "UK"          : 66435600,
     "Italy"       : 60243406,
     "Korea_South" : 51780579,
     "Spain"       : 47100396,
     "Netherlands" : 17447881,
     "Belgium"     : 11524454,
     "Sweden"      : 10333456,
     "Switzerland" : 8586550,
   }
   return pop[pays] if pays in pop else None

nfig = 0

dir_plots = "plots/"
dir_data = "/home/riw/data/COVID-19/"
ficC = dir_data + "time_series_covid19_confirmed_global.csv"
ficD = dir_data + "time_series_covid19_deaths_global.csv"

Confirmed, ett, regionsC, ctps = iol.lire_csv_tab2(ficC,lignes_entete=1,colonnes_desc=4)
Dead, ett, regionsD, ctp2 = iol.lire_csv_tab2(ficD,lignes_entete=1,colonnes_desc=4)
# pays = pays[1:]

# vecteur temps tps
ctps = ctps[4:]
if type(eval(ctps[0])) is str:
   for n in range(len(ctps)):
      ctps[n] = eval(ctps[n])
date_fin = ctps[-1].split('/')
cjour = "2020/%2.2d/%2.2d" %(int(date_fin[0]),int(date_fin[1]))
tps = [datetime(2020,1,1,12,0,0)]*len(ctps)
for n in range(len(ctps)):
   tt  = ctps[n].split('/')
   tpn = list(tps[0].timetuple())
   tpn[1], tpn[2] = int(tt[0]), int(tt[1])
   tps[n] = datetime(*tuple(tpn)[:7])

pays0C = np.ndarray((len(regionsC),),dtype='<U50')
for n in range(len(regionsC)):
   pays0C[n] = regionsC[n][1]
   if pays0C[n][0] is "'":
      pays0C[n] = eval(pays0C[n])

pays0D = np.ndarray((len(regionsD),),dtype='<U50')
for n in range(len(regionsD)):
   pays0D[n] = regionsD[n][1]
   if pays0D[n][0] is "'":
      pays0D[n] = eval(pays0D[n])

# Standards de nom
for n in range(len(pays0C)):
   if pays0C[n] == "United Kingdom": pays0C[n] = "UK"
for n in range(len(pays0D)):
   if pays0D[n] == "United Kingdom": pays0D[n] = "UK"

if len(pays0C) == Confirmed.shape[0]:
   print("Confirmed: %d regionsC rescensées" %len(pays0C))
else:
   print("inconsistance: nb de régions != nb lignes de data")
   sys.exit()

if len(pays0D) == Dead.shape[0]:
   print("Dead: %d regionsC rescensées" %len(pays0D))
else:
   print("inconsistance: nb de régions != nb lignes de data")
   sys.exit()

pays = ["France","Germany","Italy","Spain","UK","Netherlands","Switzerland",
        "Sweden","Iran","US","Korea_South","Japan","China"]
pays = ["France","Germany","Italy","Spain","UK","Netherlands",
        "Sweden","US","Japan","China"]

# Somme cummulées
csConfirmed = np.zeros((len(pays),Confirmed.shape[1]))
csDead      = np.zeros((len(pays),Dead.shape[1]))

# Somme sur tous les térritoires d'un pays
for n in range(len(pays)):
   kk = np.where(pays0C == pays[n])[0]
   for k in kk:
      csConfirmed[n] += np.nan_to_num(Confirmed[k])
   ll = np.where(pays0D == pays[n])[0]
   for l in ll:
      csDead[n] += np.nan_to_num(Dead[l])

# vecteur temps
njour = Confirmed.shape[1]
ijour = np.linspace(-njour-1,0,njour)
jjour = np.linspace(1,njour,njour)
nfilt, Fc  = 5, 0.33

for n in range(len(pays)):
   csConfirmed[n][np.where(csConfirmed[n] <= 0)[0]] = np.nan

for p in pays:
   n = np.where(np.array(pays) == p)[0][0]
   C, D = csConfirmed[n], csDead[n]
   ii  = np.where(C >= 100)[0]
   i0 = ii[0] if len(ii > 0) else 0
   tC100, C100 = jjour[i0:] -jjour[i0], C[i0:]
   jj  = np.where(D >= 10)[0]
   j0 = jj[0] if len(jj > 0) else 0
   tD10, D10 = jjour[j0:]-jjour[j0], D[j0:]
   CpM = C/population_pays(p)*1e6
   alphaC  = 100*np.nan_to_num(np.diff(np.log(C)))
   alphaCf = sig.binomialfilt1(alphaC,3)
   doublCf = np.log(2)/alphaCf*100
   DpM = D/population_pays(p)*1e6
   alphaD  = 100*np.nan_to_num(np.diff(np.log(D)))
   alphaDf = sig.binomialfilt1(alphaD,3)
   doublDf = np.log(2)/alphaDf*100
   exec(p + "= {}")
   exec(p + "['C'] = C")
   exec(p + "['tC100'] = tC100")
   exec(p + "['C100'] = C100")
   exec(p + "['CpM'] = CpM")
   exec(p + "['alphaC'] = alphaC")
   exec(p + "['alphaCf'] = alphaCf")
   exec(p + "['doublCf'] = doublCf")
   exec(p + "['D'] = D")
   exec(p + "['tD10'] = tD10")
   exec(p + "['D10'] = D10")
   exec(p + "['DpM'] = DpM")
   exec(p + "['alphaD'] = alphaD")
   exec(p + "['alphaDf'] = alphaDf")
   exec(p + "['doublDf'] = doublDf")

# time series
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.semilogy(tps,eval(p + "['C']"),lw=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc="lower left")
plt.xlabel("days")
plt.ylabel("# detected cases")
plt.title("Detected cases per million inhabitants %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_Confirmed_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                      cjour[8:10])
plt.savefig(figname + ".pdf")

# time series
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.semilogy(eval(p + "['tC100']"),eval(p + "['C100']"),lw=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc="lower right")
plt.xlabel("number of days since 100th Confirmed")
plt.ylabel("# detected cases")
plt.title("Detected cases %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_Confirmed100_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                      cjour[8:10])
plt.savefig(figname + ".pdf")

# Taux de croissance journalier
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.plot(ijour[:-1],eval(p + "['alphaC']"),lw=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc="upper left")
plt.xlabel("days")
plt.ylabel("Growth rate per day (%)")
plt.title("Growth rate of Confirmed cases %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_GR-Confirmed_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                         cjour[8:10])
plt.savefig(figname + ".pdf")

# Taux de croissance lissé
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.plot(ijour[:-1],eval(p + "['alphaCf']"),lw=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc="upper left")
plt.xlabel("days")
plt.ylabel("Growth rate per day (%)")
plt.title("COVID-19: Growth rate of Detected cases %s (smoothed)" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_GR-Confirmed-f_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                           cjour[8:10])
plt.savefig(figname + ".pdf")

# Temps de doublement,
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.plot(ijour[:-1],eval(p + "['doublCf']"),linewidth=2,label=p)
plt.gcf().autofmt_xdate()
plt.ylim((0,10))
plt.grid()
plt.legend(loc="upper left")
plt.xlabel("days")
plt.ylabel("Doubling Time (day)")
plt.title("COVID-19: Doubling Time for Confimed cases %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_GR-Confirmed-Doubling-time_%s-%s-%s" %(
                                               cjour[:4],cjour[5:7],cjour[8:10])
plt.savefig(figname + ".pdf")

nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.semilogy(tps,eval(p + "['D']"),linewidth=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc="lower left")
plt.xlabel("days")
plt.ylabel("# detected cases")
plt.title("Deaths per million inhabitants %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_Dead_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                 cjour[8:10])
plt.savefig(figname + ".pdf")

# time series
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.semilogy(eval(p + "['tD10']"),eval(p + "['D10']"),lw=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc = "lower right")
plt.xlabel("number of days since 10th Death")
plt.ylabel("# deaths")
plt.title("# Deaths   %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_Dead10_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                      cjour[8:10])
plt.savefig(figname + ".pdf")

nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.plot(ijour[:-1],eval(p + "['alphaD']"),lw=2,label=p)
plt.gcf().autofmt_xdate()
plt.grid()
plt.legend(loc="upper left")
plt.xlabel("days")
plt.ylabel("Growth rate per day (%)")
plt.title("Growth rate of Deaths cases %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_GR-Dead_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                    cjour[8:10])
plt.savefig(figname + ".pdf")

nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.plot(ijour[:-1],eval(p + "['alphaDf']"),lw=2,label=p)
plt.grid()
plt.legend(loc="upper left")
plt.xlabel("days")
plt.ylabel("Growth rate per day (%)")
plt.title("COVID-19: Growth rate of Deaths cases %s (smoothed)" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_GR-Dead-f_%s-%s-%s" %(cjour[:4],cjour[5:7],
                                                      cjour[8:10])
plt.savefig(figname + ".pdf")

# Temps de doublement,
nfig += 1
plt.figure(nfig,figsize=(12,8)); plt.clf()
for p in pays:
   lw = 2 if p == "Italy" or p == "France" else 1
   plt.plot(ijour[:-1],eval(p + "['doublDf']"),lw=2,label=p)
plt.ylim((0,10))
plt.grid()
plt.legend(loc="upper left")
plt.xlabel("days")
plt.ylabel("Doubling Time (day)")
plt.title("COVID-19: Doubling Time for Deaths cases %s" %cjour)
plt.draw()
figname = dir_plots + "COVID-19_GR-deaths-Doubling-time_%s-%s-%s" %(
                                               cjour[:4],cjour[5:7],cjour[8:10])
plt.savefig(figname + ".pdf")

plt.show(False)
