# date:    2023年11月21日 intern verison
# author： 倪成宇
# title：  初始化设置

""" 库导入 """
import pandas as pd 
import numpy as np
import pickle
from tqdm import *
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats
import statsmodels.api as sm
import os

#时间
import datetime

# 米筐
import rqdatac
from rqdatac import *
init(timeout=300)
from rqfactor_utils import *
from rqfactor_utils.universe_filter import *
from rqfactor import *
from rqfactor.extension import rolling_window, CombinedRollingWindowFactor, CombinedFactor, UserDefinedLeafFactor,UnaryCrossSectionalFactor
from rqfactor import CS_REGRESSION_RESIDUAL,MA,STD,PCT_CHANGE,REF,LOG,RANK,IF,ABS,TS_FILLNA,TS_ZSCORE,SUM,DELTA,TS_MAX,CS_ZSCORE,QUANTILE

# 关闭通知
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


# 动态券池
def INDEX_FIX(start_date = '2016-02-01',end_date = '2023-08-01',index_item = '000906.XSHG'):
    """
    :param start_date: 开始日 -> str
    :param end_date: 结束日 -> str 
    :param index_item: 指数代码 -> str 
    :return index_fix: 动态因子值 -> unstack
    """
    
    index = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in index_components(index_item,start_date= start_date,end_date=end_date).items()])).T

    # 构建动态股票池 
    index_fix = index.unstack().reset_index().iloc[:,-2:]
    index_fix.columns = ['date','stock']
    index_fix.date = pd.to_datetime(index_fix.date)
    index_fix['level'] = True
    index_fix.dropna(inplace = True)
    index_fix = index_fix.set_index(['date','stock']).level.unstack()
    index_fix.fillna(False,inplace = True)
    stock_list = index_fix.columns.tolist()

    return index_fix, stock_list


# 热力图    
def hot_corr(name,ic_df):
    """
    :param name: 因子名称 -> list 
    :param ic_df: ic序列表 -> dataframe 
    :return fig: 热力图 -> plt
    """
    ax = plt.subplots(figsize=(len(name), len(name)))#调整画布大小
    ax = sns.heatmap(ic_df[name].corr(),vmin=0.4, square=True, annot= True,cmap = 'Blues')   #annot=True 表示显示系数
    plt.title('Factors_IC_CORRELATION')
    # 设置刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)



def factor_analysis(df,name,universe,periods = 20,quantile = 10):
    """
    :param df: 因子值 -> unstack
    :param name: 因子名称 -> str
    :param periods: 预测周期(默认20) -> int 
    :param quantile: 分组(默认10) -> int
    :return result: 汇总报告 -> dict
    :return IC: ic报告 -> dataframe
    """
    result = analyse(df,start_date = df.index[0],end_date= df.index[-1],winzorization='mad',industry_neutralization = 'citics_2019',universe=universe,periods=periods,quantile = quantile,universe_filter='fundamentals',keep_preprocess_result = True)
    
    # ic分析
    corr = abs(spearmanr(pd.DataFrame(result.quantile_returns).T.cumsum().iloc[-1].rank(ascending = False).tolist(),list(np.arange(1,quantile + 1)))[0])
    
    if result.ic_values.mean() > 0:
        top_return = (result.quantile_returns[0].cumsum().tolist()[-1]+1) ** (252/len(result.quantile_returns[0]))-1
    else:
        top_return = (result.quantile_returns[quantile - 1].cumsum().tolist()[-1]+1) ** (252/len(result.quantile_returns[quantile - 1]))-1

    IC = {'name': name,
        'IC mean':round(result.ic_values.mean(),4),
        'IC std':round(result.ic_values.std(),4),
        'IR':round(result.ic_values.mean()/result.ic_values.std(),4),
        'IR_ly':round(result.ic_values[-252:].mean()/result.ic_values[-252:].std(),4),
        'IC>0':round(len(result.ic_values[result.ic_values>0].dropna())/len(result.ic_values),4),
        'ABS_IC>2%':round(len(result.ic_values[abs(result.ic_values) > 0.02].dropna())/len(result.ic_values),4),
        'Top_return':round(top_return,4),
        'Rank':round(corr,4),
        # 'AUC':round(AUC_IC(result.ic_values),4)
        }
    
    print(IC)
    IC = pd.DataFrame([IC])

    return result,IC

def create_dir_not_exist(path):
    # 若不存在该路径则自动生成
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

def quick_factor_analyse_report(factor_dict,index_item,report = True):
    """
    :param factor_dict: 因子值字典 -> dict
    :param index_item: 券池基准 -> str
    :return ic_summary: ic汇总报告 -> dataframe
    """
    ic_summary = pd.DataFrame()
    for k,v in factor_dict.items():
        try:
            result,ic_summary_temp = factor_analysis(v,k,index_item)
            ic_summary = pd.concat([ic_summary,ic_summary_temp],axis = 0)
            path = f'./dump_{index_item[:-5]}/'
            create_dir_not_exist(path)
            if report:
                pickle.dump(result,open(f'{path}{k}_{index_item[:-5]}.pkl','wb'))
            else:
                pass
        except:
            print(f'ERROR {k}')
    ic_summary.set_index(['name'],inplace = True)

    return ic_summary


def gen_report(index_item):
    """
    :param index_item: 券池 -> str
    :return NA 直接输出报告
    """
    path = f'./dump_{index_item[:-5]}/'
    files = sorted(os.listdir(path))
    files = [i for i in files if 'pkl' in i]

    result_dict = {i.replace('.pkl',''):pd.read_pickle(f'{path}{i}') for i in files}

    report_batch(
        result_dict,
        index_item,
        path,
        True)


#------------------------------------------------------------------------

# 券池过滤
def get_new_stock_filter(stock_list,date_list, newly_listed_threshold = 252):

    """
    :param stock_list: 股票队列 -> list
    :param date_list: 交易日队列 -> list
    :param newly_listed_threshold: 新股临界日 -> int
    :return df: 过滤表 -> dataframe bool
    """

    listed_date_list = [rqdatac.instruments(stock).listed_date for stock in stock_list]        
    newly_listed_window = pd.Series(index=stock_list, data=[rqdatac.get_next_trading_date(listed_date, n=newly_listed_threshold) for listed_date in listed_date_list]) 
    # 
    newly_listed_window.index.names = ['order_book_id']
    newly_listed_window = newly_listed_window.to_frame('date')
    newly_listed_window['signal'] = True
    newly_listed_window = newly_listed_window.reset_index().set_index(['date','order_book_id']).signal.unstack('order_book_id').reindex(index=date_list)
    newly_listed_window = newly_listed_window.shift(-1).bfill().fillna(False)

    print('剔除新股已构建')

    return newly_listed_window

def get_st_filter(stock_list,date_list):
    """
    :param stock_list: 股票队列 -> list
    :param date_list: 交易日队列 -> list
    :return df: 过滤表 -> dataframe bool
    """
    st_filter = rqdatac.is_st_stock(stock_list,date_list[0],date_list[-1]).reindex(columns=stock_list,index = date_list)                                #剔除ST
    st_filter = st_filter.shift(-1).fillna(method = 'ffill')
    print('剔除ST已构建')

    return st_filter

def get_suspended_filter(stock_list,date_list):
    """
    :param stock_list: 股票队列 -> list
    :param date_list: 交易日队列 -> list
    :return df: 过滤表 -> dataframe bool
    """
    suspended_filter = rqdatac.is_suspended(stock_list,date_list[0],date_list[-1]).reindex(columns=stock_list,index=date_list)
    suspended_filter = suspended_filter.shift(-1).fillna(method = 'ffill')
    print('剔除停牌已构建')

    return suspended_filter

def get_limit_up_down_filter(stock_list,date_list):
    """
    :param stock_list: 股票队列 -> list
    :param date_list: 交易日队列 -> list
    :return df: 过滤表 -> dataframe bool
    """
    # 涨停则赋值为1,反之为0    
    price = rqdatac.get_price(stock_list,date_list[0],date_list[-1],adjust_type='none',fields = ['open','limit_up'])
    df = (price['open'] == price['limit_up']).unstack('order_book_id').shift(-1).fillna(False)
    print('剔除开盘涨停已构建')

    return df

# 数据清洗函数 -----------------------------------------------------------
# MAD:中位数去极值
def mad(df):
    """
    :param df: 原始因子 -> dataframe
    :return df: 中位数去极值后因子 -> dataframe
    """
    # MAD:中位数去极值
    def filter_extreme_MAD(series,n): 
        median = series.median()
        new_median = ((series - median).abs()).median()
        return series.clip(median - n*new_median,median + n*new_median)
    # 离群值处理
    df = df.apply(lambda x :filter_extreme_MAD(x,3), axis=1)

    return df

def standardize(df):
    """
    :param df: 原始因子 -> dataframe
    :return df: 标准化后因子 -> dataframe
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)



def neutralization(df,index_item = ''):

    """
    :param df: 因子值 -> unstack
    :param df_result: 中性化后的因子值 -> unstack
    """

    order_book_ids = df.columns.tolist()
    datetime_period = df.index.tolist()
    start = datetime_period[0].strftime("%F")
    end = datetime_period[-1].strftime("%F")
    #获取行业/市值暴露度
    try:
        df_industy_market = pd.read_pickle(f'tmp/df_industy_market_{index_item}_{start}_{end}.pkl')
    except:
        market_cap = execute_factor(LOG(Factor('market_cap_3')),order_book_ids,start,end).stack().to_frame('market_cap')
        industry_df = get_industry_exposure(order_book_ids,datetime_period)
        #合并因子
        industry_df['market_cap'] = market_cap
        df_industy_market = industry_df
        df_industy_market.index.names = ['datetime','order_book_id']
        df_industy_market.dropna(inplace = True)
        create_dir_not_exist('tmp')
        df_industy_market.to_pickle(f'tmp/df_industy_market_{index_item}_{start}_{end}.pkl')

    df_industy_market['factor'] = df.stack()
    df_industy_market.dropna(subset = 'factor',inplace = True)
    
    #OLS回归
    df_result = pd.DataFrame(columns = order_book_ids,index = datetime_period)
    for i in tqdm(datetime_period):
        try:
            df_day = df_industy_market.loc[i]
            x = df_day.iloc[:,:-1]   #市值/行业
            y = df_day.iloc[:,-1]    #因子值
            df_result.loc[i] = sm.OLS(y.astype(float),x.astype(float),hasconst=False, missing='drop').fit().resid
        except:
            pass
    df_result.index.names = ['datetime']

    return df_result


def get_industry_exposure(order_book_ids,datetime_period):
    
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :return result: 虚拟变量 -> dataframe
    """
    print('gen industry martix... ')
    zx2019_industry = rqdatac.client.get_client().execute('__internal__zx2019_industry')
    df = pd.DataFrame(zx2019_industry)
    df.set_index(['order_book_id', 'start_date'], inplace=True)
    df = df['first_industry_name'].sort_index()
    
    #构建动态行业数据表格
    index = pd.MultiIndex.from_product([order_book_ids, datetime_period], names=['order_book_id', 'datetime'])
    pos = df.index.searchsorted(index, side='right') - 1
    index = index.swaplevel()   # level change (oid, datetime) --> (datetime, oid)
    result = pd.Series(df.values[pos], index=index)
    result = result.sort_index()
    
    #生成行业虚拟变量
    return pd.get_dummies(result)


def get_industry_map(order_book_ids, datetime_period):
    """
    :param order_book_ids: 股票池 -> list
    :param datetime_period: 研究日 -> list
    :return result: 个股对应行业 -> dataframe
    """
    zx2019_industry = rqdatac.client.get_client().execute('__internal__zx2019_industry')
    df = pd.DataFrame(zx2019_industry)
    df.set_index(['order_book_id', 'start_date'], inplace=True)
    df = df['first_industry_name'].sort_index()

    #构建动态行业数据表格
    index = pd.MultiIndex.from_product([order_book_ids, datetime_period], names=['order_book_id', 'datetime'])
    pos = df.index.searchsorted(index, side='right') - 1
    index = index.swaplevel()   # level change (oid, datetime) --> (datetime, oid)
    result = pd.Series(df.values[pos], index=index)
    result = result.sort_index()

    return result