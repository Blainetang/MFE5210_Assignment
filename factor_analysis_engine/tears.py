#
# Copyright 2023 Blaine TANG.
# Based on alphalens_reloaded
#
from pandas.plotting import table
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import plotting
import performance as perf
import utils
import os


class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 8))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.5, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None

    def save(self, filepath):
        """
        Save the current figure to a file.

        :param filepath: The path to save the figure to.
        """
        if self.fig is not None:
            self.fig.savefig(filepath, format='png')
        else:
            raise ValueError("No figure to save. Make sure to create a figure before calling save.")


@plotting.customize
def create_summary_tear_sheet(factor_data, group_neutral=False, demeaned=True, file_path = None, turnover_periods=None):
    # IC 分析
    ic = perf.factor_information_coefficient(factor_data, group_neutral)
    columns_wide = 2
    fr_cols = len(ic.columns)
    vertical_sections = 3 + fr_cols*3 
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)
    ic_data = ic.mean()>0
    ic_table = plotting.plot_information_table(ic)

    # 换手分析
    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True
        ).to_numpy()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(turnover_periods)
    quantile_factor = factor_data["factor_quantile"]
    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    turnover_table, auto_corr = plotting.plot_turnover_table(autocorrelation, quantile_turnover)
    # 收益率分析
    quantile_max = factor_data["factor_quantile"].max()
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=demeaned,
        group_adjust=group_neutral,
    )

    mean_quant_annual_ret = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        by_year=True,
        demeaned=demeaned,
        group_adjust=group_neutral)

    # 年化的
    mean_quant_annual_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period="252D"
    )
    # 日度的收益率均值 便于画图和计算等
    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=demeaned,
        group_adjust=group_neutral,
    )

    mean_quant_ret_bydate_not_excess = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=group_neutral,
    )[0]

    # 计算多空收益，分组收益
    long_short_info = utils.get_long_short_info(ic_data, quantile_max)
    # 超额收益
    long_portfolio_annual_transaction_cost = utils.get_long_portfolio_annual_transaction_cost(turnover_table, long_short_info)
    long_portfolio_annual_transaction_cost.name = "long_portfolio_annual_transaction_cost"
    long_portfolio_annual_transaction_cost = pd.DataFrame(long_portfolio_annual_transaction_cost).T
    long_portfolio_annual_rateret = mean_quant_annual_rateret.apply(lambda x: x.loc[long_short_info["long"].loc[x.name]])
    long_portfolio_annual_rateret.name = "long_portfolio_annual_rateret"
    long_portfolio_annual_rateret = pd.DataFrame(long_portfolio_annual_rateret).T
    factor_info_df = pd.concat([ic_table.T, turnover_table, long_portfolio_annual_transaction_cost, \
                                long_portfolio_annual_rateret], axis=0)
    info_ax = gf.next_row()
    info_ax.axis('off')
    table_ = table(info_ax, factor_info_df.round(3).reset_index(), cellLoc='center', loc="center", colWidths=[.3,.2,.2,.2])
    table_.scale(1, 2)
    factor_excess_return_info = perf.get_returns_info(mean_quant_ret_bydate, long_short_info)
    factor_return_info = perf.get_returns_info(mean_quant_ret_bydate_not_excess, long_short_info)[0]

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )
    # returns are provided.
    title = (
        "Equal Weighted "
        + ("Group Neutral " if group_neutral else "")
        + ("Long/Short ")
        + "Portfolio Cumulative Return"
    )
    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)
    plotting.plot_quantile_returns_bar(
        mean_quant_annual_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row()
    )
    plotting.plot_cumulative_returns(
        factor_excess_return_info[1], title=title, ax=gf.next_row()
    )
    ax_return_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_quantile_cumulative_returns(factor_excess_return_info[0], ax=ax_return_hqq[::2], type_="excess")
    plotting.plot_quantile_cumulative_returns(factor_return_info, ax=ax_return_hqq[1::2])
    ax_quantile_return_by_year = [gf.next_row() for x in range(fr_cols)]
    plotting.plot_quantile_returns_bar_by_year(mean_quant_annual_ret, ax=ax_quantile_return_by_year)

    plt.show()
    if file_path is not None:
        if not os.path.exists(file_path):
            # If it doesn't exist, create the directory
            os.makedirs(file_path)
        factor_data.to_parquet(os.path.join(file_path, "factor_data.parquet"))
        ic_table.to_csv(os.path.join(file_path, "ic_summary.csv"))
        gf.save(os.path.join(file_path, "report.png"))
    gf.close()


@plotting.customize
def create_returns_tear_sheet(
    factor_data, ic_data, demeaned=True, group_neutral=False, by_group=False
):
    """
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots
    by_group : bool
        If True, display graphs separately for each group.
    """
    quantile_max = factor_data["factor_quantile"].max()
    #factor_returns = perf.factor_returns(factor_data, demeaned, group_neutral,equal_weight=True)

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=demeaned,
        group_adjust=group_neutral,
    )

    mean_quant_annual_ret = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        by_year=True,
        demeaned=demeaned,
        group_adjust=group_neutral)
    # 非年化的收益率
    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )
    # 年化的
    mean_quant_annual_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period="252D"
    )
    # 日度的收益率均值 便于画图和计算等
    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=demeaned,
        group_adjust=group_neutral,
    )

    mean_quant_ret_bydate_not_excess = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=group_neutral,
    )[0]

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )
    
    # 日度收益率标准差
    compstd_quant_daily = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    # 观察多空头收益差距是否明显， 多空头收益是否稳定
    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    # 计算多空收益，分组收益
    long_short_info = utils.get_long_short_info(ic_data, quantile_max)
    # 超额收益
    factor_excess_return_info = perf.get_returns_info(mean_quant_ret_bydate, long_short_info)
    factor_return_info = perf.get_returns_info(mean_quant_ret_bydate_not_excess, long_short_info)[0]
    alpha_beta = perf.factor_alpha_beta(
        factor_data, None, demeaned, group_neutral
    )
    columns_wide = 2
    fr_cols = len(mean_quant_rateret.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    plotting.plot_returns_table(alpha_beta, mean_quant_rateret, mean_ret_spread_quant)

    plotting.plot_quantile_returns_bar(
        mean_quant_annual_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row()
    )
    plotting.plot_quantile_returns_violin(
        mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # returns are provided.
    # 以下内容需要修改
    title = (
        "Equal Weighted "
        + ("Group Neutral " if group_neutral else "")
        + ("Long/Short ")
        + "Portfolio Cumulative Return"
    )

    plotting.plot_cumulative_returns(
        factor_excess_return_info[1], title=title, ax=gf.next_row()
    )
    ax_return_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_quantile_cumulative_returns(factor_excess_return_info[0], ax=ax_return_hqq[::2], type_="excess")
    plotting.plot_quantile_cumulative_returns(factor_return_info, ax=ax_return_hqq[1::2])
    ax_quantile_return_by_year = [gf.next_row() for x in range(fr_cols)]
    plotting.plot_quantile_returns_bar_by_year(mean_quant_annual_ret, ax=ax_quantile_return_by_year)
    ax_mean_quantile_returns_spread_ts = [gf.next_row() for x in range(fr_cols)]
    plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.show()
    gf.close()


@plotting.customize
def create_information_tear_sheet(factor_data, group_neutral=False, by_group=False):
    """
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    group_neutral : bool
        Demean forward returns by group before computing IC.
    by_group : bool
        If True, display graphs separately for each group.
    """

    ic = perf.factor_information_coefficient(factor_data, group_neutral)

    plotting.plot_information_table(ic)

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    if not by_group:

        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="M",
        )
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(mean_monthly_ic, ax=ax_monthly_ic_heatmap)

    if by_group:
        mean_group_ic = perf.mean_information_coefficient(
            factor_data, group_adjust=group_neutral, by_group=True
        )

        plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row())

    plt.show()
    gf.close()
    return ic.mean()>0


@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):
    """
    Creates a tear sheet for analyzing the turnover properties of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True
        ).to_numpy()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(turnover_periods)

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    turnover_table, auto_corr = plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()
    return turnover_table


@plotting.customize
def create_full_tear_sheet(
    factor_data, long_short=True, group_neutral=False, by_group=False
):
    """
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis
    by_group : bool
        If True, display graphs separately for each group.
    """

    plotting.plot_quantile_statistics_table(factor_data)
    ic_data = create_information_tear_sheet(
        factor_data, group_neutral, by_group, set_context=False
    )
    create_returns_tear_sheet(
        factor_data,ic_data, long_short, group_neutral, by_group, set_context=False
    )
    create_turnover_tear_sheet(factor_data, set_context=False)


@plotting.customize
def create_event_returns_tear_sheet(
    factor_data,
    returns,
    avgretplot=(5, 15),
    long_short=True,
    group_neutral=False,
    std_bar=True,
    by_group=False,
):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, the factor
        quantile/bin that factor value belongs to and (optionally) the group
        the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    returns : pd.DataFrame
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after)
        If not None, plot quantile average cumulative returns
    long_short : bool
        Should this computation happen on a long short portfolio? if so then
        factor returns will be demeaned across the factor universe
    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
    std_bar : boolean, optional
        Show plots with standard deviation bars, one for each quantile
    by_group : bool
        If True, display graphs separately for each group.
    """

    before, after = avgretplot

    avg_cumulative_returns = perf.average_cumulative_return_by_quantile(
        factor_data,
        returns,
        periods_before=before,
        periods_after=after,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    num_quantiles = int(factor_data["factor_quantile"].max())

    vertical_sections = 1
    if std_bar:
        vertical_sections += ((num_quantiles - 1) // 2) + 1
    cols = 2 if num_quantiles != 1 else 1
    gf = GridFigure(rows=vertical_sections, cols=cols)
    plotting.plot_quantile_average_cumulative_return(
        avg_cumulative_returns,
        by_quantile=False,
        std_bar=False,
        ax=gf.next_row(),
    )
    if std_bar:
        ax_avg_cumulative_returns_by_q = [gf.next_cell() for _ in range(num_quantiles)]
        plotting.plot_quantile_average_cumulative_return(
            avg_cumulative_returns,
            by_quantile=True,
            std_bar=True,
            ax=ax_avg_cumulative_returns_by_q,
        )

    plt.show()
    gf.close()

    if by_group:
        groups = factor_data["group"].unique()
        num_groups = len(groups)
        vertical_sections = ((num_groups - 1) // 2) + 1
        gf = GridFigure(rows=vertical_sections, cols=2)

        avg_cumret_by_group = perf.average_cumulative_return_by_quantile(
            factor_data,
            returns,
            periods_before=before,
            periods_after=after,
            demeaned=long_short,
            group_adjust=group_neutral,
            by_group=True,
        )

        for group, avg_cumret in avg_cumret_by_group.groupby(level="group"):
            avg_cumret.index = avg_cumret.index.droplevel("group")
            plotting.plot_quantile_average_cumulative_return(
                avg_cumret,
                by_quantile=False,
                std_bar=False,
                title=group,
                ax=gf.next_cell(),
            )

        plt.show()
        gf.close()


@plotting.customize
def create_event_study_tear_sheet(
    factor_data, returns, avgretplot=(5, 15), rate_of_ret=True, n_bars=50
):
    """
    Creates an event study tear sheet for analysis of a specific event.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single event, forward returns for each
        period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
    returns : pd.DataFrame, required only if 'avgretplot' is provided
        A DataFrame indexed by date with assets in the columns containing daily
        returns.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    avgretplot: tuple (int, int) - (before, after), optional
        If not None, plot event style average cumulative returns within a
        window (pre and post event).
    rate_of_ret : bool, optional
        Display rate of return instead of simple return in 'Mean Period Wise
        Return By Factor Quantile' and 'Period Wise Return By Factor Quantile'
        plots
    n_bars : int, optional
        Number of bars in event distribution plot
    """

    long_short = False

    plotting.plot_quantile_statistics_table(factor_data)

    gf = GridFigure(rows=1, cols=1)
    plotting.plot_events_distribution(
        events=factor_data["factor"], num_bars=n_bars, ax=gf.next_row()
    )
    plt.show()
    gf.close()

    if returns is not None and avgretplot is not None:

        create_event_returns_tear_sheet(
            factor_data=factor_data,
            returns=returns,
            avgretplot=avgretplot,
            long_short=long_short,
            group_neutral=False,
            std_bar=True,
            by_group=False,
        )

    factor_returns = perf.factor_returns(factor_data, demeaned=False, equal_weight=True)

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret = mean_quant_ret.apply(
            utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
        )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False, demeaned=long_short
    )
    if rate_of_ret:
        mean_quant_ret_bydate = mean_quant_ret_bydate.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_quant_ret_bydate.columns[0],
        )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 1
    gf = GridFigure(rows=vertical_sections + 1, cols=1)

    plotting.plot_quantile_returns_bar(
        mean_quant_ret, by_group=False, ylim_percentiles=None, ax=gf.next_row()
    )

    plotting.plot_quantile_returns_violin(
        mean_quant_ret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    plt.show()
    gf.close()
