# last updated: 2021-03-15

import pandas as pd
import numpy as np
import re
import math
import json
import pickle
import platform
import webbrowser

from pandas import Timestamp as stamp

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

from urllib.request import urlopen
from IPython.display import HTML, IFrame, Image, display, display_html
from pandas.tseries.offsets import YearEnd, QuarterEnd, MonthEnd, YearBegin, QuarterBegin, MonthBegin


########################################################
# DataFrame 그래프에서 한글과 마이너스 부호 폰트 깨짐 해결 #
########################################################
# matplotlib.get_cachedir() 새로운 폰트 설치후, matplotlib의 폰트 리스트가 저장된 파일을 삭제
# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

plt.rcParams['font.family'] = 'batang'
plt.rcParams['axes.unicode_minus'] = False


########################################################
#################### Preliminary #######################
########################################################
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))


# pandas
idx = pd.IndexSlice
np.set_printoptions(precision=4)
pd.set_option("display.max_rows", 20)
# pd.set_option('display.max_columns', 500)
pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
# pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x) 
# if abs(x) < 100 and x!=np.nan else '{:,.0f}'.format(x))
pd.set_option('precision', 4)


# colors
colors = list(matplotlib.colors.cnames.keys())


# 금융안정보고서 스타일 그래프 색 지정(RGB)
lcolors = [(85/256, 142/256, 213/256), (110/256, 110/256, 110/256), 
           (175/256, 175/256, 105/256), (250/256, 192/256, 145/256), 
           (150/256, 115/256, 170/256)]
bcolors =  [(185/256, 214/256, 245/256), (191/256, 191/256, 191/256), 
            (190/256, 205/256, 150/256), (250/256, 220/256, 190/256), 
            (203/256, 185/256, 213/256)]
dcolors = lcolors + ['brown', 'orange', 'green', 'purple', 'pink', 'red', 'cyan', 'olive', 'blue']


# PLOT
# matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rc('mathtext', default='regular')
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['figure.titlesize'] = 22
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['axes.titlepad'] = 10
matplotlib.rcParams['axes.labelpad'] = 1
matplotlib.rcParams['figure.titleweight'] = 20


########################################################
#################### OPEN API ##########################
########################################################

# open ECOS API
def open_ecosapi():
    '''open ecos api web page in web browser'''
    url = ('https://ecos.bok.or.kr/jsp/openapi/OpenApiController.jsp?t=' + 
           'guideStatCd&menuGroup=MENU000004&menuCode=MENU000024')
    webbrowser.open(url)


def open_ecos():
    '''open ecos web page in web browser'''
    url = ('https://ecos.bok.or.kr')
    webbrowser.open(url)


def ecos(code = '021Y125', item1 = '?', item2 = '?', item3 = '?', 
         freq = 'Q', first = '1900', last = '2100', N = '10000', 
         detail = True, col = None):
    '''retreive monthly, quarterly, annul time series from ecos.
    run 'open_ecosapi() to explore ecos api codes.'''
    ecos_key = "http://ecos.bok.or.kr/api/StatisticSearch/390S6FIOF95M7MHASMEA"
    freq_str = {'QQ': 'Q', 'MM': '-'}
    freq += freq  # Y, Q, M, D -> YY, QQ, MM, DD
    url  = f"{ecos_key}/json/kr/1/{N}/{code}/{freq}/{first}/{last}/{item1}/{item2}/{item3}/"
    result = urlopen(url)
    data = json.loads(result.read())
    data = data["StatisticSearch"]["row"]
    df = pd.DataFrame(data)
    if detail:
        print(f"통계: {df.loc[0, 'STAT_NAME']}",
              f"단위: {df.loc[0, 'UNIT_NAME']}",
              f"기간: {df.loc[0, 'TIME']} - {df.loc[df.index[-1], 'TIME']}",
              f"항목: {df.loc[0, 'ITEM_NAME1']}",)
    df = df.set_index("TIME")
    df.index.names = ['DATE']
    if (freq == 'MM'):
        df.index = pd.DatetimeIndex(df.index.str[:4] + freq_str[freq] + df.index.str[4:])
        df.index = df.index + MonthEnd()
    elif (freq == 'QQ'):
        df.index = pd.DatetimeIndex(df.index.str[:4] + freq_str[freq] + df.index.str[4:])
        df.index = df.index + QuarterEnd()
    elif (freq == 'YY'):
        df.index = pd.DatetimeIndex(df.index)
        df.index = df.index + YearEnd()
    elif (freq == 'DD'):
        df.index = pd.DatetimeIndex(df.index)
    else:
        print('frequency is not one of D, M, Q, A.')
        return
    df["DATA_VALUE"] = df["DATA_VALUE"].astype("float")
    return df['DATA_VALUE'].to_frame(col)


### FRED ###
# fred = Fred(api_key='3ea8df05757a52a97e5db46423e0cf13')


########################################################
################ Funcations ############################
########################################################
dp4 = lambda x: '{:,.4f}'.format(x)
dp3 = lambda x: '{:,.3f}'.format(x)
dp2 = lambda x: '{:,.2f}'.format(x)
dp1 = lambda x: '{:,.1f}'.format(x)
dp0 = lambda x: '{:,.0f}'.format(x)



def multi2single(df, sep = ':'):
    '''expand singleindex to multiindex with : separator'''
    if df.columns.nlevels == 1:
        pass
    else:
        df.columns = [sep.join(col).strip() for col in df.columns.values]
    return df


def single2multi(df, sep = ':'):
    '''collapse multiindex to single index with : separator'''
    df.columns = pd.MultiIndex.from_tuples(df.columns.str.split(sep).tolist())
    return df


def sptitles(ax, titles, y = 1.03):
    '''new version of set_titles: even when lengths of ax and titles are different, this works'''
    axs = ax.ravel()
    n1 = len(axs)
    n2 = len(titles)
    if n1 > n2:
        [titles.append('') for i in range(len(axs) - len(titles))]
    titles = titles[:len(axs)]
    try:
        [axe.set_title(titles[i], y=y) for i, axe in enumerate(ax.ravel())]
    except:
        ax.set_title(titles, y=y)


def smfmt(x, p):
    return '{:,.2f}'.format(x) if abs(x) < 100 else '{:,.0f}'.format(x)


# def normalize(ts):
#     return (ts - ts.mean())/ts.std()


def scatter_label(ax, df, fontsize=12, is_datetime=True, text=''):
    for k, v in df.iterrows():
        if is_datetime:
            k = text + f"{k:%y}"
        else:
            k = text
        ax.annotate(k, v, xytext=(0, 0), textcoords='offset points',
                    family='sans-serif', fontsize=fontsize, color='black')


def save2xlsx(df, fname, fmt='%Y-%m'):
    '''Datetimeindex is converted to str with fmt: %Y %B %m %d %r'''
    df1 = df.copy()
    df1.index = df1.index.strftime(fmt)
    df1.to_excel('output//' + fname)


# def set_titles(ax, titles, fontsize=18, y=1.03):
#     try:
#         [axe.set_title(titles[i], fontsize=fontsize, y=y) for i, axe in enumerate(ax.ravel())]
#     except:
#         ax.set_title(titles, fontsize=fontsize, y=y)


# def set_hline(ax, y=0, color='k', alpha=0.5, lw=3, linestyle='--'):
#     try:
#         [axe.axhline(y=y, color=color, alpha=alpha, lw=lw, linestyle=linestyle) for axe in ax.ravel()]
#     except:
#         ax.axhline(y=y, color=color, alpha=alpha, lw=lw, linestyle=linestyle)



##################        EXAMPLE       ##############################################
######################################################################################
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
######################################################################################
######################################################################################

# df.plot(ax=ax)
# ticklabels = get_ticklabels(index, 1, '%y')
# ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
def get_ticklabels(index, i, fmt = '%Y'):
    # fmt = '%y', '%Q\n%Y', ...
    ticklabels = [''] * len(index)
    ticklabels[::i] = [item.strftime(fmt) for item in index[::i]]
    return ticklabels


def xaxis_date_format(ax, fmt = '%y'):
    '''fmt: '%b\n%y', '%y-%m-%d', etc '''
    # plt.draw() must be run before this func is called.
    # ax.get_xticklabels() is not populated until figure is drawn
    ticklabels = [pd.to_datetime(t.get_text()).strftime(fmt) for t in ax.get_xticklabels()]
    ax.set_xticklabels(ticklabels, rotation=0)


# ax.plot(df.index, df)
# ax.xaxis_year_format(ax, 1, '%y')

# convert date objects from pandas format to python datetime
# df.index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in df.index]
def xaxis_year_locate_format(ax, i = 2, fmt = '%y'):
    ax.xaxis.set_major_locator(mdates.YearLocator(i))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def yaxis_format(ax, i = 1):
    string = '{:,.' + str(i) + 'f}'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: string.format(x)))


def xaxis_format(ax, i = 1):
    string = '{:,.' + str(i) + 'f}'
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: string.format(x)))


def trim_legend(ax, p0=1, p1=3, ncol=3):
    ax.legend([i.get_text()[p0:p1] for i in ax.legend().get_texts()[:]], ncol=ncol)


def display_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


