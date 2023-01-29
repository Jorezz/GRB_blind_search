import requests
from bs4 import BeautifulSoup

def parse_datetime_from_response_text(text, time_tag):
    ob = BeautifulSoup(text,'lxml')
    category = ob.find_all("td", {"id" : time_tag})
    return ' '.join(category[0].string.split())

def get_utc_from_ijd(ijd_time):
    url = f'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl?time_in_i=&time_in_c=&time_in_d=&time_in_j=&time_in_m=&time_in_sf=&time_in_wf=&time_in_ii={ijd_time}&time_in_sl=&time_in_sni=&time_in_snu=&time_in_s=&time_in_h=&time_in_sz=&time_in_ss=&time_in_sn=&timesys_in=u&timesys_out=u&apply_clock_offset=yes'
    response = requests.get(url)
    return parse_datetime_from_response_text(response.text, "time_out_i")

def get_ijd_from_utc(utc_time) -> float:
    utc_time = utc_time[:10]+'+'+utc_time[11:13]+'%3A'+utc_time[14:16]+'%3A'+utc_time[17:19]
    url = f'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl?time_in_i={utc_time}&time_in_c=&time_in_d=&time_in_j=&time_in_m=&time_in_sf=&time_in_wf=&time_in_ii=&time_in_sl=&time_in_sni=&time_in_snu=&time_in_s=&time_in_h=&time_in_sz=&time_in_ss=&time_in_sn=&timesys_in=u&timesys_out=u&apply_clock_offset=yes'
    response = requests.get(url)
    return float(parse_datetime_from_response_text(response.text, "time_out_ii"))

def get_utc_from_Fermi_seconds(fermi_seconds):
    url = f'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl?time_in_i=&time_in_c=&time_in_d=&time_in_j=&time_in_m=&time_in_sf={fermi_seconds}&time_in_wf=&time_in_ii=&time_in_sl=&time_in_sni=&time_in_snu=&time_in_s=&time_in_h=&time_in_sz=&time_in_ss=&time_in_sn=&timesys_in=u&timesys_out=u&apply_clock_offset=yes'
    response = requests.get(url)
    return parse_datetime_from_response_text(response.text, "time_out_i")

def get_ijd_from_Fermi_seconds(fermi_seconds) -> float:
    utc_time = utc_time[:10]+'+'+utc_time[11:13]+'%3A'+utc_time[14:16]+'%3A'+utc_time[17:19]
    url = f'https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl?time_in_i=&time_in_c=&time_in_d=&time_in_j=&time_in_m=&time_in_sf={fermi_seconds}&time_in_wf=&time_in_ii=&time_in_sl=&time_in_sni=&time_in_snu=&time_in_s=&time_in_h=&time_in_sz=&time_in_ss=&time_in_sn=&timesys_in=u&timesys_out=u&apply_clock_offset=yes'
    response = requests.get(url)
    return float(parse_datetime_from_response_text(response.text, "time_out_ii"))