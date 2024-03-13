import pandas as pd
import rqdatac as rq
from datetime import datetime
import os

rq.init()

class Data_processing:

    def __init__(self, type_) -> None:
        """
        初始化
        Parameters
        ----------
        type_ : str
            选取的是主板股票"main"还是所有股票"all"
        """
        self.today_ = str(datetime.today())[:10]
        self.type_ = type_
        self.stk_codes = []
        self.last_date = ''

    @staticmethod
    def create_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def get_stocks_info(self):
        """
        获取所有股票信息
        """
        stock_df = rq.all_instruments(type="CS", date=None, market="cn")
        stock_df = stock_df[stock_df["status"]!="Unknown"]
        stock_df["days_after_listed"] = stock_df["listed_date"].apply(lambda x: \
                                                                      (pd.to_datetime(self.today_) - pd.to_datetime(x)).days)
        if self.type_ == "main":
            stock_df = stock_df[stock_df["board_type"]=="MainBoard"]
        info_path = f".{self.type_}/data/stock_info"
        self.create_directory(info_path)
        stock_df.to_pickle(info_path + "/all_stks.pkl")
        self.stk_codes = list(stock_df["order_book_id"])
        suspended_df = rq.is_suspended(self.stk_codes, start_date="2019-01-01", end_date=self.today_)
        st_df = rq.is_st_stock(self.stk_codes, start_date="2019-01-01", end_date=self.today_)
        suspended_df.to_pickle(info_path + "/suspended_df.pkl")
        st_df.to_pickle(info_path + "/st_df.pkl")

    def get_daily_price_data(self):
        """
        获取日频价格数据
        """
        daily_data_path = f".{self.type_}/data/price_data/daily"
        self.create_directory(daily_data_path)
        file_path = daily_data_path+"/daily_data.parquet"
        def compare_pre_and_now(pre_df, now_df):
            """
            判断期间价格是否发生复权，若发生复权则更新所有价格数据
            Parameters
            ----------
            pre_df : dataframe
                之前的价格df
            now_df : dataframe
                新价格df

            Returns
            -------
            tuple
                (发生复权的股票代码, 更新后的价格df)
            """
            pre_df = pd.pivot(pre_df[pre_df["date"]==self.last_date], index="date", columns="order_book_id", values="close")
            now_df = pd.pivot(now_df[now_df["date"]==self.last_date], index="date", columns="order_book_id", values="close")
            if_match = (pre_df == now_df[pre_df.columns])
            not_match_list = list(if_match[if_match==False].dropna(axis=1).columns)
            if len(not_match_list)>0:
                not_match_prices = rq.get_price(not_match_list, "2019-01-01", self.today_, adjust_type="pre_volume").reset_index()
                return not_match_list, not_match_prices
            else:
                return None
        
        if os.path.isfile(file_path):
            last_df = pd.read_parquet(file_path)
            self.last_date = last_df["date"].sort_values().iloc[-1]
            temp_daily_price = rq.get_price(self.stk_codes, self.last_date, self.today_, adjust_type="pre_volume")
            temp_daily_price["vwap"] = temp_daily_price["total_turnover"] / temp_daily_price["volume"]
            temp_daily_price = temp_daily_price.reset_index()
            daily_price = pd.concat([last_df,temp_daily_price]).sort_values(["order_book_id", "date"]).drop_duplicates()
            if compare_pre_and_now(last_df, temp_daily_price) is None:
                pass
            else:
                not_match_list, not_match_prices = compare_pre_and_now(last_df, temp_daily_price)
                daily_price = daily_price[daily_price["order_book_id"].apply(lambda x: x not in not_match_list)]
                daily_price = pd.concat([daily_price,not_match_prices]).sort_values(["order_book_id", "date"]).drop_duplicates()
        else:
            daily_price = rq.get_price(self.stk_codes, "2019-01-01", self.today_, adjust_type="pre_volume")
            daily_price["vwap"] = daily_price["total_turnover"] / daily_price["volume"]
            daily_price = daily_price.reset_index()
        daily_price.to_parquet(file_path)
    
    def get_turnover_rate(self):
        """
        获取换手率数据
        """
        daily_data_path = f".{self.type_}/data/price_data/daily"
        self.create_directory(daily_data_path)
        file_path = daily_data_path+"/turnover_rate.parquet"
        if os.path.isfile(file_path):
            last_rate_df = pd.read_parquet(file_path)
            try:
                temp_rate_df = rq.get_turnover_rate(self.stk_codes, self.last_date, self.today_).reset_index()
                turnover_rate_df = pd.concat([last_rate_df,temp_rate_df]).sort_values(["order_book_id", "tradedate"]).drop_duplicates()
            except AttributeError:
                print("换手率数据未更新,稍后再试!")
        else:
            turnover_rate_df = rq.get_turnover_rate(self.stk_codes, "2019-01-01", self.today_).reset_index()
        try:
            turnover_rate_df.to_parquet(file_path)
        except UnboundLocalError:
            print("换手率数据未更新,稍后再试!")

    def get_factor_exposure(self):
        """
        得到barra风格暴露度,之后做风格中性化或者风控时会用到
        """
        barra_path = f".{self.type_}/data/barra"
        self.create_directory(barra_path)
        file_path = barra_path+"/factor_exposure.parquet"
        if os.path.isfile(file_path):
            last_barra_df = pd.read_parquet(file_path)
            try:
                temp_barra_df = rq.get_factor_exposure(self.stk_codes, self.last_date, self.today_, factors = None,industry_mapping='sws_2021', model = 'v1').reset_index()
                barra_factr_df = pd.concat([last_barra_df,temp_barra_df]).sort_values(["order_book_id", "date"]).drop_duplicates()
            except AttributeError:
                print("BARRA数据未更新,稍后再试!")
        else:
            barra_factr_df = rq.get_factor_exposure(self.stk_codes, "2019-01-01", self.today_, factors = None,industry_mapping='sws_2021', model = 'v1').reset_index()
        try:
            barra_factr_df.to_parquet(file_path)
        except UnboundLocalError:
            print("BARRA数据未更新,稍后再试!")
    
    def get_minibar_data(self):
        """
        获取分钟线数据
        """
        path = f".{self.type_}/data/minibar"
        last_minibar_date = sorted(os.listdir(path))[-1][:10]
        if os.path.exists(path):
            trading_days =  rq.get_trading_dates(last_minibar_date, self.today_)
        else:
            os.makedirs(path)
            trading_days =  rq.get_trading_dates("2019-01-01", self.today_)
        for date in trading_days:
            minibar_data = rq.get_price(self.stk_codes, str(date), str(date), frequency="1m", adjust_type="pre_volume").reset_index()
            minibar_data.to_parquet(path+f"/{str(date)}.parquet")
        
    def get_index_data(self):
        """
        获取指数信息, 现在有中证全指和中证500
        """
        index_path = f".{self.type_}/data/index_info"
        self.create_directory(index_path)
        zzqz = rq.get_price("000985.XSHG", start_date="2019-01-01", end_date=self.today_, adjust_type="pre").reset_index()
        zz500 = rq.get_price("000905.XSHG", start_date="2019-01-01", end_date=self.today_, adjust_type="pre").reset_index()
        zzqz.to_pickle(index_path+"/zzqz_info.pkl")
        zz500.to_pickle(index_path+"/zz500_info.pkl")


    def auto(self):
        self.get_stocks_info()
        self.get_daily_price_data()
        self.get_turnover_rate()
        self.get_factor_exposure()
        self.get_minibar_data()
        self.get_index_data()

if __name__ == "__main__":
    dd = Data_processing("all")
    dd.auto()

