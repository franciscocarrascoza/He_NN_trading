// FIX: ChartWidget header for OHLCV candlestick chart with prediction overlays per spec

#ifndef CHARTWIDGET_H  // FIX: header guard
#define CHARTWIDGET_H

#include <QWidget>  // FIX: Qt widget base class
#include <QtCharts/QChart>  // FIX: Qt Charts
#include <QtCharts/QChartView>  // FIX: Qt Chart view
#include <QtCharts/QCandlestickSeries>  // FIX: candlestick series for OHLCV
#include <QtCharts/QLineSeries>  // FIX: line series for conformal bands
#include <QtCharts/QScatterSeries>  // FIX: scatter series for prediction arrows

class ChartWidget : public QWidget  // FIX: ChartWidget class per spec
{
    Q_OBJECT  // FIX: Qt meta-object macro

public:
    explicit ChartWidget(QWidget *parent = nullptr);  // FIX: constructor
    ~ChartWidget();  // FIX: destructor

public slots:
    void updateChart(const QJsonObject &data);  // FIX: update chart with new data
    void addPredictionOverlay(double timestamp, double mu, double lower, double upper);  // FIX: add prediction arrow and conformal band

private:
    void setupChart();  // FIX: initialize chart components

    QtCharts::QChartView *chartView;  // FIX: chart view pointer
    QtCharts::QChart *chart;  // FIX: chart pointer
    QtCharts::QCandlestickSeries *candlestickSeries;  // FIX: candlestick series for OHLCV per spec
    QtCharts::QScatterSeries *predictionSeries;  // FIX: scatter series for prediction arrows per spec
    QtCharts::QLineSeries *conformalLowerSeries;  // FIX: line series for lower conformal band per spec
    QtCharts::QLineSeries *conformalUpperSeries;  // FIX: line series for upper conformal band per spec
};

#endif  // FIX: close header guard
