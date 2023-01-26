import requests
from bs4 import BeautifulSoup

def get_utc_from_ijd(ijd_time):
    url = f'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl?time_in_i=&time_in_c=&time_in_d=&time_in_j=&time_in_m=&time_in_sf=&time_in_wf=&time_in_ii={ijd_time}&time_in_sl=&time_in_sni=&time_in_snu=&time_in_s=&time_in_h=&time_in_sz=&time_in_ss=&time_in_sn=&timesys_in=u&timesys_out=u&apply_clock_offset=yes'
    response = requests.get(url)
    ob = BeautifulSoup(response.text,'lxml')
    category = ob.find_all("td", {"id" : "time_out_i"})
    return category[0].string.split()[0]+' '+category[0].string.split()[1]

def get_ijd_from_utc(utc_time):
    utc_time = utc_time[:10]+'+'+utc_time[11:13]+'%3A'+utc_time[14:16]+'%3A'+utc_time[17:19]
    url = f'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl?time_in_i={utc_time}&time_in_c=&time_in_d=&time_in_j=&time_in_m=&time_in_sf=&time_in_wf=&time_in_ii=&time_in_sl=&time_in_sni=&time_in_snu=&time_in_s=&time_in_h=&time_in_sz=&time_in_ss=&time_in_sn=&timesys_in=u&timesys_out=u&apply_clock_offset=yes'
    response = requests.get(url)
    ob = BeautifulSoup(response.text,'lxml')
    category = ob.find_all("td", {"id" : "time_out_ii"})
    return category[0].string.split()[0]