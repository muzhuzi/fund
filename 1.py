# 20260416 yaoyingming 修改 交易日至期末的涨跌幅 逻辑，由期初期末的直接除，改为连乘
#                      修改 增加到数据库的逻辑
# 20260417 yaoyingming 修改权益经理获取逻辑， 权益经理到每天，方便后续处理。
#                      修改追涨杀跌指数逻辑，修改交易日前5天涨跌幅数据的逻辑。
# 20260514 yaoyingming 1. 修改mport的逻辑，mport_base_data中先对数据进行4倍笛卡尔积（之前是用pd.concat(8倍)，速度较慢），之后在mport_data中进行全行业和申万行业的 pd.concat
#                      2. 修改入库方法由execute many 改为copy_expert,速度提升明显
# 20260518 yaoyingming 1. 对于多组合 avg_nav avg_stk_mktval由之前的先avg再sum改为日内sum，然后求avg
#                      2. 合并单组合中全行业 和 申万行业 2个方法为1个，简化流程。
#                      3. 修正投资经理期间的from to 闭区间为左闭右开
# 20260527 yaoyingming 1. 增加内存占用的打印信息

import numpy as np
import hashlib
import pandas as pd
import gc
import psutil,os,time

from work.dao.crs.DBOperation import *
from work.common.LogUtil import log_config;logger = log_config()
SYS_PARM_DEFAULT = {}

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/1024/1024

class DataOperation():
    def __init__(self, sk_date, days):
        self.dbo = DBOperation()
        self.end_sk_date = pd.to_datetime(sk_date)
        self.get_data_from_db(sk_date)
        self.days = days
        self.start_sk_date = self.end_sk_date - pd.Timedelta(days = self.days)
        self.current_sk_date = self.end_sk_date
        self.gen_date_dim()
        logger.info(f'Param start_sk_date: {self.start_sk_date},end_sk_date: {self.end_sk_date}, and total days: {days}')

    def gen_date_dim(self):
        self.date_dim = []
        for i in range(self.days):
            # logger.debug(i)
            sk_date = (self.end_sk_date - pd.Timedelta(days=i)).strftime('%Y%m%d')

            end_date = pd.Timestamp(sk_date)
            ytd_start_date = end_date.replace(month=1, day=1)
            ytd_pre_start_date = ytd_start_date - pd.Timedelta(days=1)

            mtd_start_date = end_date.replace(day=1)
            mtd_pre_start_date = mtd_start_date - pd.Timedelta(days=1)

            l1m_start_date = end_date - pd.DateOffset(months=1)
            l1m_pre_start_date = l1m_start_date - pd.Timedelta(days=1)
            # 今年以来
            # 本月以来
            # 近一月
            self.date_dim.append({'sk_date': sk_date, 'dims': [
                {'date_dim': '今年以来', 'start_date': ytd_start_date, 'end_date': end_date,
                 'pre_start_date': ytd_pre_start_date},
                {'date_dim': '本月以来', 'start_date': mtd_start_date, 'end_date': end_date,
                 'pre_start_date': mtd_pre_start_date},
                {'date_dim': '近一月', 'start_date': l1m_start_date, 'end_date': end_date,
                 'pre_start_date': l1m_pre_start_date}
            ]})
        logger.debug(self.date_dim)
        return self.date_dim

    def get_data_from_db(self, sk_date):
        # 默认返回400 天的数据
        # 交易信息
        self.trd_fa_inv_tran_df = self.dbo.get_ods_trd_fa_inv_tran(sk_date)
        logger.debug(self.trd_fa_inv_tran_df)
        if len(self.trd_fa_inv_tran_df) == 0:
            logger.debug('no trade data found!')
            quit()

        # 组合净值规模信息
        self.fact_port_nav_df = self.dbo.get_fact_port_nav(sk_date)
        logger.debug(self.fact_port_nav_df)

        # 股票日均市值
        self.bi_risk_tb_df = self.dbo.get_bi_risk_tb(sk_date)
        logger.debug(self.bi_risk_tb_df)

        # 权益投资经理信息： 到每日的经理信息
        self.prod_fund_assoc_invpty_df = self.dbo.get_prod_fund_assoc_invpty()
        logger.debug(self.prod_fund_assoc_invpty_df)

        # 前5个工作日
        self.dim_time_b5days_df = self.dbo.get_dim_time_b5days()
        logger.debug(self.dim_time_b5days_df)

        # 股市信息
        self.issu_stock_quotation_df = self.dbo.get_issu_stock_quotation(sk_date)
        logger.debug(self.issu_stock_quotation_df)

        self.trd_fa_inv_tran_df = self.add_b5days_quote_chg(self.trd_fa_inv_tran_df, self.issu_stock_quotation_df)

    def gen_hashed_hex(self, name):
        if not isinstance(name, str):
            name = str(name)
        # "根据mport_name进行编码 ：
        #  hex(md5(mport_name))%90000000 + 10000000"
        md5_hex_str = hashlib.md5(name.encode('utf-8')).hexdigest()
        # md5_int = int(md5_hex_str, 16)
        md5_int = int(md5_hex_str.encode('utf-8').hex())
        calc_val = (md5_int % 90000000) + 10000000
        return calc_val

    def add_b5days_quote_chg(self, trd_fa_inv_tran_df, issu_stock_quotation_df):
        quotation_df = issu_stock_quotation_df.rename(columns={'sk_date': 'sk_date2'})
        ## 股票每日的前5日涨跌幅上
        main_df = trd_fa_inv_tran_df[['sk_date', 'serial_no', 'deal_price', 'sk_issue']]
        main_df = pd.merge(main_df,
                           self.dim_time_b5days_df,
                           on='sk_date', how='left')
        logger.debug(main_df)

        # 数据数倍膨胀
        main_df1 = pd.merge(main_df, quotation_df, on='sk_issue', how='left')

        # 只保留交易后的数据
        data_mask = ((main_df1['sk_date2'] <= main_df1['sk_date']) & (main_df1['sk_date2'] >= main_df1['b5days']))
        # main_df1 = main_df1[data_mask].copy()
        main_df1 = main_df1[data_mask].copy()

        # main_df1['b5days_quote_chg'] = np.where(
        #     main_df1['sk_date'] == main_df1['sk_date2'],
        #     main_df1['price_close'] / main_df1['deal_price'],  # 交易日当天用成交价格
        #     main_df1['price_close'] / main_df1['last_price_close']
        # )

        conditions = [(main_df1['deal_price'] == 0),
                      (main_df1['last_price_close'] == 0),
                      main_df1['sk_date'] == main_df1['sk_date2'],
                      main_df1['sk_date'] != main_df1['sk_date2']]
        choices = [0, 0,main_df1['price_close'] / main_df1['deal_price'],main_df1['price_close'] / main_df1['last_price_close']]
        main_df1['b5days_quote_chg'] = np.select(conditions, choices,default=0)

        group_by_cols = ['sk_date', 'serial_no', 'sk_issue', 'deal_price']
        # 与main_df相比，增加一列quote_chg 每笔交易到统计期末的涨跌幅，记录数无变化
        main_df1 = main_df1.groupby(group_by_cols)['b5days_quote_chg'].prod().reset_index()

        main_df1['b5days_quote_chg'] = main_df1['b5days_quote_chg'] - 1
        main_df1 = pd.merge(trd_fa_inv_tran_df, main_df1[['serial_no', 'b5days_quote_chg']], on='serial_no', how='left')
        # logger.debug(main_df1.info(memory_usage='deep'))
        return main_df1

    def get_mng_name_end_date(self, end_date):
        # 获取权益投资经理的信息
        # 只取最后的end_date的权益经理
        data_mask = (self.prod_fund_assoc_invpty_df['sk_date'] == end_date)
        df = self.prod_fund_assoc_invpty_df[data_mask].copy()

        # logger.debug(df.info(memory_usage='deep'))
        return df[['sk_portfolio', 'mng_name']]

    def get_bi_risk_tb_df(self, pre_start_date, end_date):
        # 股票日均市值
        data_mask = ((self.bi_risk_tb_df['sk_date'] >= pre_start_date) & (self.bi_risk_tb_df['sk_date'] <= end_date))
        bi_risk_tb_df = self.bi_risk_tb_df[data_mask].copy()
        bi_risk_tb_df = bi_risk_tb_df.groupby('sk_portfolio', as_index=False).agg(avg_stk_mktval=('stk_mktval', 'mean'))
        logger.debug(bi_risk_tb_df)
        return bi_risk_tb_df

    def get_fact_port_nav_df(self, pre_start_date, end_date):
        # 先过滤再排序：
        # 统计期间的开始和结束时间 过滤：
        data_mask = ((self.fact_port_nav_df['sk_date'] >= pre_start_date) & (
                    self.fact_port_nav_df['sk_date'] <= end_date))  # 按有规模的天数计算，含起始日期前一日
        fact_port_nav_df = self.fact_port_nav_df[data_mask].sort_values(['sk_portfolio', 'sk_date']).copy()

        # 统计期间的日均规模
        port_avg_nav = fact_port_nav_df.groupby(['sk_portfolio'], as_index=False).agg(
            avg_nav=('net_assets', 'mean')).copy()

        # 计算累计净值 + 累计天数
        fact_port_nav_df['cumsum_net_assets'] = fact_port_nav_df.groupby('sk_portfolio')['net_assets'].cumsum()
        fact_port_nav_df['cum_days'] = fact_port_nav_df.groupby('sk_portfolio').cumcount() + 1
        # fact_port_nav_df['last_cumsum_net_assets'] =

        # 得到期末累计值
        fact_port_nav_df_end = fact_port_nav_df[(fact_port_nav_df['sk_date'] == end_date)]
        fact_port_nav_df_end = fact_port_nav_df_end[['sk_portfolio', 'cumsum_net_assets', 'cum_days']]
        # fact_port_nav_df_end = fact_port_nav_df_end.drop('net_assets',axis=1)
        fact_port_nav_df_end.rename(columns={'cumsum_net_assets': 'end_cumsum_net_assets', 'cum_days': 'end_cum_days'},
                                    inplace=True)

        fact_port_nav_df = pd.merge(fact_port_nav_df, fact_port_nav_df_end, on='sk_portfolio', how='left')
        # 每日到期末的日均规模计算
        fact_port_nav_df['avg_txednav'] = ((fact_port_nav_df['end_cumsum_net_assets'] - fact_port_nav_df[
            'cumsum_net_assets'] + fact_port_nav_df['net_assets'])
                                           / (fact_port_nav_df['end_cum_days'] - fact_port_nav_df['cum_days'] + 1))
        logger.debug(fact_port_nav_df)

        # main_df = pd.merge(main_df,fact_port_nav_df[['sk_portfolio','sk_date','avg_txednav']],left_on=['sk_portfolio','bus_date'],right_on=['sk_portfolio','sk_date'],how='left')
        return fact_port_nav_df, port_avg_nav

    def get_issu_stock_quotation_df(self):
        # 开始日期前5天的数据 到结束日期的数据
        # pre_start_date = start_date - pd.Timedelta(days=7)
        # data_mask = ((self.issu_stock_quotation_df['sk_date'] <= end_date) & (self.issu_stock_quotation_df['sk_date'] >= pre_start_date))
        # issu_stock_quotation_df = self.issu_stock_quotation_df[data_mask].copy()  # 筛选截止日期前的数据
        issu_stock_quotation_df = self.issu_stock_quotation_df.copy()  # 筛选截止日期前的数据
        # last_issu_stock_quotation_df  = issu_stock_quotation_df.sort_values('sk_date').groupby('sk_issue').tail(1)
        return issu_stock_quotation_df

    def data_process_by_sk_issue(self, trades_df):
        # logger.debug(main_df1.columns)
        # trades_df.to_excel(r'd:\c1.xlsx', index=False)
        trades_df['avg_txednav'] = trades_df['avg_txednav'].fillna(0)
        ## 转换为个股粒度：
        result = (trades_df.assign(
            # 买入金额 * 交易日至期末的涨跌幅
            buy_contri=np.where(trades_df['avg_txednav'] == 0,
                                0,
                                trades_df['buy_amt'] * trades_df['quote_chg'] / trades_df['avg_txednav']),
            # 卖出金额 * 交易日至期末的涨跌幅
            sell_contri=np.where(trades_df['avg_txednav'] == 0,
                                 0,
                                 trades_df['sell_amt'] * trades_df['quote_chg'] / trades_df['avg_txednav']),
            # ∑(买入金额*前5日的涨跌幅/区间买入金额)
            chs_amt=trades_df['buy_amt'] * trades_df['b5days_quote_chg'],
            # ∑(买入金额*前5日的涨跌幅/区间买入金额)
            kll_amt=trades_df['sell_amt'] * trades_df['b5days_quote_chg']
        )
        .groupby(['sk_portfolio', 'sk_issue', 'bk_portfolio', 'portfolio_name', 'prod_type1_name', 'inds_name_lv1',
                  'mng_name', 'avg_nav', 'avg_stk_mktval'], as_index=False, dropna=False)
        .agg(
            trd_amt=('trade_amount', 'sum'),
            buy_amt=('buy_amt', 'sum'),
            sell_amt=('sell_amt', 'sum'),
            buy_contri=('buy_contri', 'sum'),
            sell_contri=('sell_contri', 'sum'),
            chs_amt=('chs_amt', 'sum'),
            kll_amt=('kll_amt', 'sum')
        )
        )

        result['tx_contri'] = result['buy_contri'] - result['sell_contri']
        result['win_tx_amt'] = np.where(result['tx_contri'] > 0, result['trd_amt'], 0)
        # ,sum(t1.trd_amt/nullif(a3.avg_stk_mktval,0)) as bila_turnover       --双边换手率
        result['bila_turnover'] = result['trd_amt'] / result['avg_stk_mktval']
        # ,sum(t1.trd_amt/nullif(a2.avg_nav,0)) as bila_na_turnover    --双边换手率(占净值)
        result['bila_na_turnover'] = result['trd_amt'] / result['avg_nav']
        result['win_stk'] = np.where(result['tx_contri'] > 0, result['sk_issue'], np.nan)

        return result

    def get_quote_chg(self, main_df, issu_stock_quotation_df):
        # 股票价格数据：
        issu_stock_quotation_df.rename(columns={'sk_date': 'sk_date2'}, inplace=True)
        ##计算 交易日至期末的涨跌幅
        # logger.debug(main_df.info(memory_usage='deep'))
        # logger.debug(issu_stock_quotation_df.info(memory_usage='deep'))
        # 数据膨胀数倍
        main_df1 = pd.merge(main_df, issu_stock_quotation_df, on=['sk_issue'], how='left')
        # logger.debug(main_df1.info(memory_usage='deep'))
        # logger.debug(main_df1.columns)
        # 只保留交易后的数据
        data_mask = (main_df1['sk_date'] <= main_df1['sk_date2'])
        main_df1 = main_df1[data_mask].copy()
        # logger.debug(main_df1)
        # main_df1['quote_chg'] = np.where(
        #     main_df1['sk_date'] == main_df1['sk_date2'],
        #     main_df1['price_close'] / main_df1['deal_price'],  # 交易日当天用成交价格
        #     main_df1['price_close'] / main_df1['last_price_close']
        # )

        conditions = [(main_df1['deal_price'] == 0),
                      (main_df1['last_price_close'] == 0),
                      main_df1['sk_date'] == main_df1['sk_date2'],
                      main_df1['sk_date'] != main_df1['sk_date2']]
        choices = [0, 0,main_df1['price_close'] / main_df1['deal_price'],main_df1['price_close'] / main_df1['last_price_close']]
        main_df1['quote_chg'] = np.select(conditions, choices,default=0)

        # main_df1[main_df1['serial_no'] == '5348436471'].to_excel(r'd:\b1.xlsx', index=False)

        group_by_cols = ['sk_date', 'serial_no', 'sk_issue', 'deal_price']
        # 与main_df相比，增加一列quote_chg 每笔交易到统计期末的涨跌幅，记录数无变化
        main_df1 = main_df1.groupby(group_by_cols)['quote_chg'].prod().reset_index()
        main_df1['quote_chg'] = main_df1['quote_chg'] - 1

        # logger.debug(main_df1.info(memory_usage='deep'))
        main_df1 = main_df1[['serial_no', 'quote_chg']]
        # logger.debug(main_df1.info(memory_usage='deep'))
        return main_df1

    def get_base_port_data(self, result, date_dim):
        # start_date = date_dim['start_date']
        result['inds_name_lv1'] = result['inds_name_lv1'].astype(str).fillna('其他行业')
        end_date = date_dim['end_date']
        # pre_start_date = date_dim['pre_start_date']
        date_dim_ = date_dim['date_dim']
        # result.to_excel(r'd:\a1.xlsx', index=False)

        all_port_sets = [['sk_portfolio', 'bk_portfolio', 'portfolio_name', 'mng_name', 'prod_type1_name', 'avg_nav','avg_stk_mktval'] # 全行业
                        ,['sk_portfolio', 'bk_portfolio', 'portfolio_name', 'mng_name', 'prod_type1_name', 'avg_nav','avg_stk_mktval', 'inds_name_lv1'] #申万行业
                         ]

        port_result = pd.concat([result.groupby(s, as_index=False, dropna=False)
        .agg(
            trd_amt=('trd_amt', 'sum'),
            buy_amt=('buy_amt', 'sum'),
            sell_amt=('sell_amt', 'sum'),
            buy_contri=('buy_contri', 'sum'),
            sell_contri=('sell_contri', 'sum'),
            tx_contri=('tx_contri', 'sum'),
            win_tx_amt=('win_tx_amt', 'sum'),  # win交易金额
            tx_stk_cnt=('sk_issue', 'nunique'),  # 交易股票个数
            win_stk_cnt=('win_stk', 'nunique'),  # 盈利股票个数
            chs_amt=('chs_amt', 'sum'),
            kll_amt=('kll_amt', 'sum'),
            bila_turnover=('bila_turnover', 'sum'),  # 双边换手率	bila_turnover
            bila_na_turnover=('bila_na_turnover', 'sum')  # 双边换手率(占净值)  bila_na_turnover
        )
            for s in all_port_sets]
        )
        port_result['date_dim'] = date_dim_
        port_result['sk_date'] = end_date

        # port_result.to_excel(r'd:\a2.xlsx', index=False)

        port_result['inds_type'] = np.where(port_result['inds_name_lv1'].isna()
                                             ,'全行业'
                                             ,'申万行业'
                                             )

        port_result['kll_index'] = np.where(port_result['sell_amt']==0, 0 ,port_result['kll_amt'] / port_result['sell_amt'])
        port_result['chs_index'] = np.where(port_result['buy_amt']==0, 0 ,port_result['chs_amt'] / port_result['buy_amt'])

        # -- 单边换手率 - -    min(买入金额, 卖出金额) / 日均股票持仓市值
        # ,case when sum(a1.buy_amt) <= sum(a1.sell_amt) then sum(a1.buy_amt) else sum(a1.sell_amt) end/max(nullif(a1.avg_stk_mktval,0)) as unil_turnover
        port_result['unil_turnover'] = np.where(port_result['buy_amt'] <= port_result['sell_amt'],
                                                  port_result['buy_amt'], port_result['sell_amt']) / port_result[
                                             'avg_stk_mktval']

        # -- 单边换手率(占净值) - -min(买入金额, 卖出金额) / 组合日均规模
        # ,case when sum(a1.buy_amt) <= sum(a1.sell_amt) then sum(a1.buy_amt) else sum(a1.sell_amt) end/max(nullif(a1.avg_nav,0))  as unil_na_turnover
        port_result['unil_na_turnover'] = np.where(port_result['buy_amt'] <= port_result['sell_amt'],
                                                     port_result['buy_amt'], port_result['sell_amt']) / \
                                            port_result['avg_nav']

        port_result['tx_win_pct'] = port_result['win_tx_amt'] / port_result['trd_amt']

        port_result.rename(columns={'mng_name': 'equ_mngr_name','trd_amt':'tx_amt', 'avg_nav': 'avg_net_assets'}, inplace=True)

        # 计算 交易金额占比	tx_amt_ratio	numeric(32,8)	单行业交易金额占总交易金额比
        # 获取行业总交易额
        ins_sum_df = port_result[port_result['inds_type'] == '全行业'][['date_dim', 'sk_portfolio', 'tx_amt']].copy()
        ins_sum_df.rename(columns={'tx_amt': 'ins_tx_amt'}, inplace=True)
        result_merge = pd.merge(port_result, ins_sum_df, on=['date_dim', 'sk_portfolio'], how='left')
        logger.debug(ins_sum_df)

        result_merge['tx_amt_ratio'] = result_merge['tx_amt'] / result_merge['ins_tx_amt']
        logger.debug(result_merge)
        # logger.debug(al_ins_result.columns)

        return result_merge

    def get_mport_base_data(self, trades_df, date_dim):
        # start_date = date_dim['start_date']
        end_date = date_dim['end_date']
        # pre_start_date = date_dim['pre_start_date']
        date_dim_ = date_dim['date_dim']
        # 过滤养老金数据
        trades_df = trades_df[trades_df['prod_type1_name'].isin(['年金', '职业年金', '养老金'])].copy()
        ## 去掉原有到end_date的权益经理信息，增加到每日的权益经理信息
        trades_df.drop(columns=['mng_name','avg_nav','avg_stk_mktval'], inplace=True)
        #补充到每个日期的mng_name
        trades_df = pd.merge(trades_df, self.prod_fund_assoc_invpty_df, on=['sk_date', 'sk_portfolio'], how='left')
        logger.debug(trades_df)
        #补充每日组合规模
        trades_df = pd.merge(trades_df, self.fact_port_nav_df, on=['sk_date', 'sk_portfolio'], how='left') # net_assets
        logger.debug(trades_df)
        #补充每日股票持仓市值
        trades_df = pd.merge(trades_df, self.bi_risk_tb_df, on=['sk_date', 'sk_portfolio'], how='left') # stk_mktval
        logger.debug(trades_df)
        trades_df['bus_date'] = trades_df['sk_date']
        # trades_df.to_excel(r'd:\d1.xlsx', index=False)
        conditions = [(trades_df['prod_type1_name'] == '年金').fillna(False).astype(bool),
                      (trades_df['prod_type1_name'] == '职业年金').fillna(False).astype(bool),
                      (trades_df['prod_type1_name'] == '养老金').fillna(False).astype(bool)]
        choices = ['全企业年金组合', '全职业年金组合', '养老金']
        trades_df['port_dim'] = np.select(conditions, choices, default='other')
        trades_df['sk_date'] = end_date
        trades_df['inds_name_lv1'] = trades_df['inds_name_lv1'].astype(str).fillna('其他行业')
        ## 明细数据过滤 年金', '职业年金', '养老金 后的集合
        # trades_df.to_excel(r'd:\d2.xlsx', index=False)
        logger.debug(trades_df)

        ####### 对明细数据cube
        port_dim_level1_df = pd.DataFrame(
            {'port_dim_level1': ['全年金组合', '全年金组合-组合类型', '全年金投资经理', '全年金投资经理-组合类型']})
        port_dim_level1_df['_tmp_key'] = 1
        logger.debug(port_dim_level1_df)
        trades_df['_tmp_key'] = 1
        trades_df['inds_type'] = '申万行业'
        logger.debug(trades_df)
        cube_trades_df = pd.merge(trades_df, port_dim_level1_df, on='_tmp_key').drop(columns='_tmp_key')
        logger.debug(cube_trades_df)

        conditions = [(cube_trades_df['port_dim_level1'] == '全年金组合'),
                      (cube_trades_df['port_dim'] == '全企业年金组合') & (cube_trades_df['port_dim_level1'] == '全年金组合-组合类型'),
                      (cube_trades_df['port_dim'] == '全职业年金组合') & (cube_trades_df['port_dim_level1'] == '全年金组合-组合类型'),
                      (cube_trades_df['port_dim'] == '养老金') & (cube_trades_df['port_dim_level1'] == '全年金组合-组合类型'),
                      (cube_trades_df['port_dim_level1'] == '全年金投资经理'),
                      (cube_trades_df['port_dim'] == '全企业年金组合') & (cube_trades_df['port_dim_level1'] == '全年金投资经理-组合类型'),
                      (cube_trades_df['port_dim'] == '全职业年金组合') & (cube_trades_df['port_dim_level1'] == '全年金投资经理-组合类型'),
                      (cube_trades_df['port_dim'] == '养老金') & (cube_trades_df['port_dim_level1'] == '全年金投资经理-组合类型')]
        choices1 = ['年金合并组合-年金汇总',
                    '年金合并组合-职业年金',
                    '年金合并组合-企业年金',
                    '年金合并组合-养老金',
                    '年金合并组合-' + cube_trades_df['mng_name'],
                    '职业年金合并组合-' + cube_trades_df['mng_name'],
                    '企业年金合并组合-' + cube_trades_df['mng_name'],
                    '养老金-' + cube_trades_df['mng_name']
                    ]
        choices2 = ['全年金组合',
                    '全职业年金组合',
                    '全企业年金组合',
                    '年金合并组合-养老金',
                    '全年金投资经理',
                    '全职业年金投资经理',
                    '全企业年金投资经理',
                    '养老金投资经理'
                    ]
        cube_trades_df['mport_name'] = np.select(conditions, choices1, default='other')
        cube_trades_df['port_dim'] = np.select(conditions, choices2, default='other')
        cube_trades_df = cube_trades_df[cube_trades_df['port_dim'] != '年金合并组合-养老金']
        # cube_trades_df.loc[cube_trades_df['port_dim'].isin(['全年金组合','全职业年金组合','全企业年金组合']),'mng_name'] = None
        ###################################################

        ##### 获取mport的 avg_txednav ################################
        avg_txednav_df = cube_trades_df[['bus_date', 'mport_name', 'avg_txednav']].drop_duplicates()
        logger.debug(avg_txednav_df)

        cube_avg_txednav = (avg_txednav_df.groupby(['bus_date', 'mport_name'], as_index=False, dropna=False)
        .agg(
            avg_txednav=('avg_txednav', 'sum'),
        ))

        avg_df = cube_trades_df[['bus_date', 'sk_portfolio', 'mport_name', 'net_assets', 'stk_mktval']].drop_duplicates()
        # avg_df.to_excel(r'd:\d1.xlsx', index=False)

        mport_avg = (avg_df.groupby(['bus_date','mport_name'], as_index=False, dropna=False) # 多组合到每日的汇总
         .agg(
            sum_nav=('net_assets', 'sum'),
            sum_stk_mktval=('stk_mktval', 'sum'),
        ))
        # mport_avg.to_excel(r'd:\d2.xlsx', index=False)
        # 多组合 日均股票持仓市值 组合日均规模
        self.mport_avg2 = (mport_avg.groupby('mport_name', as_index=False, dropna=False).agg(
            avg_nav=('sum_nav', 'mean'),
            avg_stk_mktval=('sum_stk_mktval', 'mean'),
        ))
        # self.mport_avg2.to_excel(r'd:\d3.xlsx', index=False)

        logger.debug(cube_avg_txednav)
        # cube_trades_df.to_excel(r'd:\d5.xlsx', index=False)
        logger.debug(trades_df)
        # cube_avg_txednav.to_excel(r'd:\d6.xlsx', index=False)
        cube_trades_df.drop(columns=['avg_txednav'], inplace=True)
        cube_trades_df1 = pd.merge(cube_trades_df, cube_avg_txednav[['mport_name','bus_date','avg_txednav']],on=['bus_date','mport_name'], how='left')
        logger.debug(cube_trades_df1)
        # logger.info(cube_trades_df1.shape)
        # cube_trades_df1.to_excel(r'd:\d7.xlsx', index=False)
        ##################################

        # 粒度改变： 组合 + 区间段维度 + 个股 + 投资经理
        cube_result = (cube_trades_df1.assign(
            # 买入金额 * 交易日至期末的涨跌幅
            buy_contri=np.where(cube_trades_df1['avg_txednav'] == 0,
                                0,
                                cube_trades_df1['buy_amt'] * cube_trades_df1['quote_chg'] / cube_trades_df1['avg_txednav']),
            # 卖出金额 * 交易日至期末的涨跌幅
            sell_contri=np.where(cube_trades_df1['avg_txednav'] == 0,
                                 0,
                                 cube_trades_df1['sell_amt'] * cube_trades_df1['quote_chg'] / cube_trades_df1['avg_txednav']),
            # ∑(买入金额*前5日的涨跌幅/区间买入金额)
            chs_amt=cube_trades_df1['buy_amt'] * cube_trades_df1['b5days_quote_chg'],
            # ∑(买入金额*前5日的涨跌幅/区间买入金额)
            kll_amt=cube_trades_df1['sell_amt'] * cube_trades_df1['b5days_quote_chg']
        )
        .groupby(['sk_date','mport_name','sk_portfolio','port_dim', 'sk_issue', 'bk_portfolio', 'portfolio_name', 'prod_type1_name', 'inds_name_lv1','inds_type'], as_index=False, dropna=False)
        .agg(
            trd_amt=('trade_amount', 'sum'),
            buy_amt=('buy_amt', 'sum'),
            sell_amt=('sell_amt', 'sum'),
            buy_contri=('buy_contri', 'sum'),
            sell_contri=('sell_contri', 'sum'),
            chs_amt=('chs_amt', 'sum'),
            kll_amt=('kll_amt', 'sum')
        )
        )

        cube_result['tx_contri'] = cube_result['buy_contri'] - cube_result['sell_contri']
        cube_result['win_tx_amt'] = np.where(cube_result['tx_contri'] > 0, cube_result['trd_amt'], 0)
        cube_result['win_stk'] = np.where(cube_result['tx_contri'] > 0, cube_result['sk_issue'], np.nan)
        cube_result['date_dim'] = date_dim_
        # cube_result.to_excel(r'd:\c1.xlsx', index=False)

        logger.debug(cube_result)
        return cube_result

    def get_mport_data(self, df, date_dim):
        cube_result = self.get_mport_base_data(df, date_dim)
        # cube_result.to_excel(r'd:\e1.xlsx', index=False)
        # 汇总到mport_name粒度，准备输出到最终表
        all_mport_sets = [['sk_date', 'port_dim', 'mport_name', 'date_dim', 'inds_type', 'inds_name_lv1'] # 申万行业
                        ,['sk_date', 'port_dim', 'mport_name', 'date_dim'] # 全行业
                          ]
        cube_result2 = pd.concat([cube_result.groupby(s, as_index=False, dropna=False)
        .agg(
            trd_amt=('trd_amt', 'sum'),
            buy_amt=('buy_amt', 'sum'),
            sell_amt=('sell_amt', 'sum'),
            buy_contri=('buy_contri', 'sum'),
            sell_contri=('sell_contri', 'sum'),
            tx_contri=('tx_contri', 'sum'),
            win_tx_amt=('win_tx_amt', 'sum'),  # win交易金额
            tx_stk_cnt=('sk_issue', 'nunique'),  # 交易股票个数
            win_stk_cnt=('win_stk', 'nunique'),  # 盈利股票个数
            chs_amt=('chs_amt', 'sum'),
            kll_amt=('kll_amt', 'sum'),
            # avg_nav=('avg_nav', 'sum'),
            # avg_stk_mktval=('avg_stk_mktval', 'sum'),
            # bila_turnover=('bila_turnover', 'sum'),  # 双边换手率	bila_turnover
            # bila_na_turnover=('bila_na_turnover', 'sum')  # 双边换手率(占净值)  bila_na_turnover
        )
        for s in all_mport_sets])

        logger.debug(cube_result2)
        # cube_result2.to_excel(r'd:\e2.xlsx', index=False)
        #####################################
        # 取原始数据进行去重后，进行汇总

        cube_result2 = pd.merge(cube_result2,self.mport_avg2[['mport_name','avg_nav','avg_stk_mktval']],on='mport_name',how='left')
        logger.debug(cube_result2)
        # cube_result2.to_excel(r'd:\e6.xlsx', index=False)

        ####################################
        # 根据mport_name进行编码 ：hex(md5(mport_name)) % 90000000 + 10000000
        cube_result2['inds_type'] = cube_result2['inds_type'].fillna('全行业')
        cube_result2['sk_mport'] = cube_result2['mport_name'].apply(self.gen_hashed_hex)
        cube_result2['date_dim'] = date_dim['date_dim']
        cube_result2['kll_index'] = np.where(cube_result2['sell_amt']==0, 0 ,cube_result2['kll_amt'] / cube_result2['sell_amt'])
        cube_result2['chs_index'] = np.where(cube_result2['buy_amt']==0, 0 ,cube_result2['chs_amt'] / cube_result2['buy_amt'])
        cube_result2['tx_win_pct'] = cube_result2['win_tx_amt'] / cube_result2['trd_amt']
        # 单边换手率
        cube_result2['unil_turnover'] = np.where(cube_result2['buy_amt'] <= cube_result2['sell_amt'],
                                                cube_result2['buy_amt'], cube_result2['sell_amt']) / cube_result2['avg_stk_mktval']
        # 单边换手率(占净值)
        cube_result2['unil_na_turnover'] = np.where(cube_result2['buy_amt'] <= cube_result2['sell_amt'],
                                                   cube_result2['buy_amt'], cube_result2['sell_amt']) / cube_result2['avg_nav']
        # ,sum(t1.trd_amt/nullif(a3.avg_stk_mktval,0)) as bila_turnover       --双边换手率
        cube_result2['bila_turnover'] = np.where(cube_result2['avg_stk_mktval']==0, 0 ,cube_result2['trd_amt'] / cube_result2['avg_stk_mktval'])
        # ,sum(t1.trd_amt/nullif(a2.avg_nav,0)) as bila_na_turnover    --双边换手率(占净值)
        cube_result2['bila_na_turnover'] = np.where(cube_result2['avg_nav']==0, 0 ,cube_result2['trd_amt'] / cube_result2['avg_nav'])
        cube_result2.rename(columns={'trd_amt': 'tx_amt', 'avg_nav': 'avg_net_assets'}, inplace=True)

        # 计算 交易金额占比	tx_amt_ratio	numeric(32,8)	单行业交易金额占总交易金额比
        # 获取行业总交易额
        ins_sum_df = (cube_result2[cube_result2['inds_type'] == '全行业']
                      .groupby(['sk_mport'], as_index=False, dropna=False)
                      .agg(ins_tx_amt=('tx_amt', 'sum')))
        logger.debug(ins_sum_df)
        cube_result2 = pd.merge(cube_result2, ins_sum_df, on='sk_mport', how='left')

        cube_result2['tx_amt_ratio'] = cube_result2['tx_amt'] / cube_result2['ins_tx_amt']
        cube_result2['inds_name_lv1'] = np.where(cube_result2['inds_type']=='全行业'
                                                 ,None
                                                 ,cube_result2['inds_name_lv1']
                                                 )

        cube_result2['memo'] = None
        cube_result2['dk_system_of_upd'] = None
        cube_result2['batchno'] = None
        cube_result2['inserttime'] = datetime.now()
        cube_result2['updatetime'] = datetime.now()
        # cube_result2.to_excel(r'd:\e7.xlsx', index=False)

        new_order = ['sk_date', 'date_dim', 'port_dim', 'mport_name', 'sk_mport'
            , 'inds_type', 'inds_name_lv1', 'avg_net_assets', 'buy_amt', 'sell_amt'
            , 'tx_amt', 'buy_contri', 'sell_contri', 'tx_contri', 'win_tx_amt'
            , 'tx_win_pct', 'tx_stk_cnt', 'win_stk_cnt', 'chs_index', 'kll_index'
            , 'unil_turnover', 'bila_turnover', 'unil_na_turnover', 'bila_na_turnover', 'tx_amt_ratio'
            , 'memo', 'dk_system_of_upd','batchno','inserttime','updatetime']
        cube_result3 = cube_result2[new_order]
        logger.debug(cube_result3)
        return cube_result3


    def get_trade_data(self, main_df, date_dim):
        # 对一个统计期间的计算
        logger.debug(date_dim)
        # logger.debug(main_df.info(memory_usage='deep'))
        start_date = date_dim['start_date']
        end_date = date_dim['end_date']
        pre_start_date = date_dim['pre_start_date']

        data_mask = ((main_df['sk_date'] <= self.current_sk_date) & (main_df['sk_date'] >= start_date))
        logger.debug(main_df)

        main_df = main_df[data_mask]
        logger.debug(main_df)

        # 权益基金经理信息
        mng_name_info_df = self.get_mng_name_end_date(end_date)
        main_df1 = pd.merge(main_df, mng_name_info_df, on=['sk_portfolio'], how='left')
        logger.debug(main_df1)

        # 组合规模
        fact_port_nav_df, port_avg_nav = self.get_fact_port_nav_df(pre_start_date, end_date)
        logger.debug(main_df1)

        # (fact_port_nav_df, port_avg_nav)

        # 每日到期末日均规模
        main_df1 = pd.merge(main_df1, fact_port_nav_df[['sk_portfolio', 'sk_date', 'avg_txednav']],
                            on=['sk_portfolio', 'sk_date'], how='left')
        logger.debug(main_df1)

        ##补充日均规模数据
        main_df1 = pd.merge(main_df1, port_avg_nav, on=['sk_portfolio'], how='left')
        logger.debug(main_df1)

        # 获得股票持仓日均规模
        bi_risk_tb_df = self.get_bi_risk_tb_df(pre_start_date, end_date)
        ##补充股票持仓日均规模
        main_df1 = pd.merge(main_df1, bi_risk_tb_df, on=['sk_portfolio'], how='left')
        logger.debug(main_df1)
        return main_df1

    def get_port_data(self, main_df, date_dim):
        # 生成 dm.fact_port_stk_tx_contri 数据
        result_merge = self.get_base_port_data(main_df, date_dim)

        # result_merge['buy_contri']=0
        # result_merge['sell_contri']=0
        # result_merge['tx_contri']=0
        result_merge['memo'] = None
        result_merge['dk_system_of_upd'] = None
        result_merge['batchno'] = None
        result_merge['inserttime'] = datetime.now()
        result_merge['updatetime'] = datetime.now()

        new_order = ['sk_date', 'date_dim', 'sk_portfolio', 'bk_portfolio', 'portfolio_name', 'equ_mngr_name',
                     'prod_type1_name', 'inds_type', 'inds_name_lv1', 'avg_net_assets', 'buy_amt', 'sell_amt', 'tx_amt',
                     'buy_contri', 'sell_contri', 'tx_contri', 'win_tx_amt', 'tx_win_pct', 'tx_stk_cnt', 'win_stk_cnt',
                     'chs_index', 'kll_index', 'unil_turnover', 'bila_turnover', 'unil_na_turnover', 'bila_na_turnover',
                     'tx_amt_ratio','memo','dk_system_of_upd','batchno','inserttime','updatetime']
        result_merge = result_merge[new_order]

        logger.debug(result_merge)
        return result_merge

    def data_process_for_one_dim(self, main_df, date_dim):
        # 得到交易粒度的数据集合
        main_df1 = self.get_trade_data(main_df, date_dim)
        # logger.debug(main_df1.columns)
        # 改变为个股粒度
        result = self.data_process_by_sk_issue(main_df1)
        logger.debug(result)

        result_port = self.get_port_data(result, date_dim)

        # 生成 表dm.fact_mport_stk_tx_contri 数据
        result_mport = self.get_mport_data(main_df1, date_dim)
        logger.debug(result_mport)
        # result_mport.to_excel(r'd:\b1.xlsx', index=False)

        # logger.debug(result.columns)
        return result_port, result_mport

    def data_process_for_one_day(self):
        # 一个sk_date 可以有多个维度的数据 ytd mtd 1m
        result2 = []
        result2_mport = []
        logger.info(f'Current sk_date: {self.current_sk_date}!')
        # 根据 date dim 筛选交易数据
        data_mask = (self.trd_fa_inv_tran_df['sk_date'] <= self.current_sk_date)
        main_df = self.trd_fa_inv_tran_df[data_mask].copy()
        logger.debug(main_df)

        # 股市股票信息
        issu_stock_quotation_df = self.get_issu_stock_quotation_df()
        # 增加期末的数据
        tmp_df = main_df[['sk_date', 'serial_no', 'sk_issue', 'deal_price']]
        quote_chg_df = self.get_quote_chg(tmp_df, issu_stock_quotation_df)
        logger.debug(quote_chg_df)
        # b5days_quote_chg_df = self.get_b5days_quote_chg(tmp_df,issu_stock_quotation_df)
        # logger.debug(b5days_quote_chg_df)

        main_df1 = pd.merge(main_df, quote_chg_df, on='serial_no', how='left')
        # main_df1 = pd.merge(main_df1,b5days_quote_chg_df,on = 'serial_no' , how='left')

        # 三个统计期间的循环
        for date_ in self.current_date_dims:
            result, result_mport = self.data_process_for_one_dim(main_df1, date_)
            result2.append(result)
            result2_mport.append(result_mport)

        result2 = pd.concat(result2)

        result2_mport = pd.concat(result2_mport)
        result2['sk_date'] = result2['sk_date'].dt.strftime('%Y%m%d')
        result2_mport['sk_date'] = result2_mport['sk_date'].dt.strftime('%Y%m%d')

        # result.to_excel(r'd:\port1.xlsx', index=False)
        # result2_mport.to_excel(r'd:\mport1.xlsx', index=False)

        # adsinvr.fact_port_stk_tx_contri
        # logger.info(result2.shape)
        # self.dbo.save_fact_port_stk_tx_contri(result2)
        # # adsinvr.fact_mport_stk_tx_contri
        # logger.info(result2_mport.shape)
        # self.dbo.save_fact_mport_stk_tx_contri(result2_mport)

        gc.collect()
        logger.info(f'fact_port_stk_tx_contri: {result2.shape}')
        logger.info(f'fact_mport_stk_tx_contri: {result2_mport.shape}')
        logger.info(f'current pid: {os.getpid()} memory used: {get_memory_usage()} MB')
        return result2, result2_mport

    def for_muti_days(self):
        # 多个sk_date的数据
        result_port_list = []
        result_mport_list = []
        i=0
        for current_date_dims in self.date_dim:
            logger.debug(current_date_dims)
            self.current_sk_date = current_date_dims['sk_date']
            self.current_date_dims = current_date_dims['dims']
            result_port,result_mport  = self.data_process_for_one_day()
            result_port_list.append(result_port)
            result_mport_list.append(result_mport)
            i=i+1
            if i % 10 == 0:
                # adsinvr.fact_port_stk_tx_contri
                result_port = pd.concat(result_port_list)
                logger.info(result_port.shape)
                self.dbo.save_fact_port_stk_tx_contri(result_port)
                result_port_list = []

                # adsinvr.fact_mport_stk_tx_contri
                result_mport = pd.concat(result_mport_list)
                logger.info(result_mport.shape)
                self.dbo.save_fact_mport_stk_tx_contri(result_mport)
                result_mport_list = []

        if len(result_port_list)!=0:
            # adsinvr.fact_port_stk_tx_contri
            result_port = pd.concat(result_port_list)
            logger.info(result_port.shape)
            self.dbo.save_fact_port_stk_tx_contri(result_port)

            # adsinvr.fact_mport_stk_tx_contri
            result_mport = pd.concat(result_mport_list)
            logger.info(result_mport.shape)
            self.dbo.save_fact_mport_stk_tx_contri(result_mport)

        # result_port_list = pd.concat(result_port_list)
        # return result_port_list

def FACT_PORT_MPORT_STK_TX_CONTRI(SYS_PARM=SYS_PARM_DEFAULT, INDEX_PARM=[], L_RETURN_COLUMNS = None):
    sk_date = (datetime.now() + timedelta(days=-1)).strftime('%Y%m%d')
    days =40
    # sk_date = '20240301'
    do = DataOperation(sk_date, days)

    ## 个股粒度的数据
    # result = do.data_process()
    result = do.for_muti_days()
    # result.to_excel(r'd:\a1.xlsx',index=False)
    # logger.debug(sys.version)
    return result

if __name__ == '__main__':
    # sk_date = (datetime.now() + timedelta(days=-1)).strftime('%Y%m%d')
    sk_date = '20240604'
    days = 1
    do = DataOperation(sk_date, days)
    logger.info(f'Param sk_date: {sk_date}, and before days: {days}')
    ## 个股粒度的数据
    # result = do.data_process()
    result = do.for_muti_days()
    # result.to_excel(r'd:\a1.xlsx',index=False)
    # logger.debug(sys.ver
