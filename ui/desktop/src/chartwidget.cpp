// FIX: ChartWidget implementation for OHLCV candlestick chart with overlays per spec

#include "chartwidget.h"  // FIX: chart widget header
#include <QVBoxLayout>  // FIX: vertical layout
#include <QtCharts/QDateTimeAxis>  // FIX: datetime axis for timestamps
#include <QtCharts/QValueAxis>  // FIX: value axis for prices

ChartWidget::ChartWidget(QWidget *parent)  // FIX: constructor implementation
    : QWidget(parent)  // FIX: call base constructor
    , chartView(nullptr)  // FIX: initialize chart view pointer
    , chart(nullptr)  // FIX: initialize chart pointer
    , candlestickSeries(nullptr)  // FIX: initialize candlestick series pointer
    , predictionSeries(nullptr)  // FIX: initialize prediction series pointer
    , conformalLowerSeries(nullptr)  // FIX: initialize lower conformal series pointer
    , conformalUpperSeries(nullptr)  // FIX: initialize upper conformal series pointer
{
    setupChart();  // FIX: initialize chart components
}

ChartWidget::~ChartWidget()  // FIX: destructor implementation
{
    // FIX: Qt will auto-delete child objects
}

void ChartWidget::setupChart()  // FIX: chart initialization implementation
{
    // FIX: Create chart
    chart = new QtCharts::QChart();  // FIX: instantiate chart
    chart->setTitle("BTCUSDT 1h with Predictions");  // FIX: set chart title per spec
    chart->setAnimationOptions(QtCharts::QChart::NoAnimation);  // FIX: disable animations for performance

    // FIX: Create candlestick series for OHLCV per spec
    candlestickSeries = new QtCharts::QCandlestickSeries();  // FIX: instantiate candlestick series
    candlestickSeries->setName("OHLCV");  // FIX: set series name
    candlestickSeries->setIncreasingColor(QColor(0, 255, 0));  // FIX: green for up candles per spec
    candlestickSeries->setDecreasingColor(QColor(255, 0, 0));  // FIX: red for down candles per spec
    chart->addSeries(candlestickSeries);  // FIX: add series to chart

    // FIX: Create prediction scatter series per spec
    predictionSeries = new QtCharts::QScatterSeries();  // FIX: instantiate scatter series
    predictionSeries->setName("Predictions (μ)");  // FIX: set series name
    predictionSeries->setColor(QColor(255, 170, 0));  // FIX: orange for predictions per spec
    predictionSeries->setMarkerSize(8.0);  // FIX: marker size
    chart->addSeries(predictionSeries);  // FIX: add series to chart

    // FIX: Create conformal interval series per spec
    conformalLowerSeries = new QtCharts::QLineSeries();  // FIX: instantiate lower line series
    conformalLowerSeries->setName("Conformal Lower");  // FIX: set series name
    conformalLowerSeries->setColor(QColor(51, 153, 255, 128));  // FIX: semi-transparent blue per spec
    chart->addSeries(conformalLowerSeries);  // FIX: add series to chart

    conformalUpperSeries = new QtCharts::QLineSeries();  // FIX: instantiate upper line series
    conformalUpperSeries->setName("Conformal Upper");  // FIX: set series name
    conformalUpperSeries->setColor(QColor(51, 153, 255, 128));  // FIX: semi-transparent blue per spec
    chart->addSeries(conformalUpperSeries);  // FIX: add series to chart

    // FIX: Create axes
    QtCharts::QDateTimeAxis *axisX = new QtCharts::QDateTimeAxis();  // FIX: datetime x-axis
    axisX->setFormat("yyyy-MM-dd hh:mm");  // FIX: datetime format
    axisX->setTitleText("Time");  // FIX: x-axis title
    chart->addAxis(axisX, Qt::AlignBottom);  // FIX: add bottom axis
    candlestickSeries->attachAxis(axisX);  // FIX: attach to candlestick series
    predictionSeries->attachAxis(axisX);  // FIX: attach to prediction series
    conformalLowerSeries->attachAxis(axisX);  // FIX: attach to lower conformal series
    conformalUpperSeries->attachAxis(axisX);  // FIX: attach to upper conformal series

    QtCharts::QValueAxis *axisY = new QtCharts::QValueAxis();  // FIX: value y-axis
    axisY->setTitleText("Price (USD)");  // FIX: y-axis title
    chart->addAxis(axisY, Qt::AlignLeft);  // FIX: add left axis
    candlestickSeries->attachAxis(axisY);  // FIX: attach to candlestick series
    predictionSeries->attachAxis(axisY);  // FIX: attach to prediction series
    conformalLowerSeries->attachAxis(axisY);  // FIX: attach to lower conformal series
    conformalUpperSeries->attachAxis(axisY);  // FIX: attach to upper conformal series

    // FIX: Create chart view
    chartView = new QtCharts::QChartView(chart, this);  // FIX: instantiate chart view
    chartView->setRenderHint(QPainter::Antialiasing);  // FIX: enable antialiasing

    // FIX: Set layout
    QVBoxLayout *layout = new QVBoxLayout(this);  // FIX: create vertical layout
    layout->addWidget(chartView);  // FIX: add chart view to layout
    layout->setContentsMargins(0, 0, 0, 0);  // FIX: remove margins
}

void ChartWidget::updateChart(const QJsonObject &data)  // FIX: update chart with new data
{
    // FIX: Skeleton implementation - would parse OHLCV data and update series
    // FIX: Full implementation would parse JSON, create QCandlestickSet objects, and append to series
    Q_UNUSED(data);  // FIX: suppress unused warning in skeleton
}

void ChartWidget::addPredictionOverlay(double timestamp, double mu, double lower, double upper)  // FIX: add prediction overlay
{
    // FIX: Add prediction point to scatter series per spec
    predictionSeries->append(timestamp, mu);  // FIX: append prediction point

    // FIX: Add conformal interval points per spec
    conformalLowerSeries->append(timestamp, lower);  // FIX: append lower bound
    conformalUpperSeries->append(timestamp, upper);  // FIX: append upper bound
}
