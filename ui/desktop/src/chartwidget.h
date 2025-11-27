// FIX: ChartWidget header for OHLCV candlestick chart with prediction overlays per spec

#ifndef CHARTWIDGET_H  // FIX: header guard
#define CHARTWIDGET_H

#include <QWidget>  // FIX: Qt widget base class
#include <QJsonObject>  // FIX: JSON object
#include <QJsonArray>  // FIX: JSON array
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
    void updateChart(const QJsonObject &data);  // FIX: update chart with new data (full replace)
    void updateCandles(const QJsonArray &candles);  // FIX: update chart with candle array (append mode)
    void addPredictionOverlay(double timestamp, double mu, double lower, double upper);  // FIX: add prediction arrow and conformal band

public:
    void updateChartTitle(const QString &symbol, const QString &timeframe);  // FIX: update chart title
    void exportToPNG();  // FIX: export chart to PNG file
    void exportToCSV();  // FIX: export chart data to CSV file

private:
    void setupChart();  // FIX: initialize chart components

    QChartView *chartView;  // FIX: chart view pointer
    QChart *chart;  // FIX: chart pointer
    QCandlestickSeries *candlestickSeries;  // FIX: candlestick series for OHLCV per spec
    QScatterSeries *predictionSeries;  // FIX: scatter series for prediction arrows per spec
    QLineSeries *conformalLowerSeries;  // FIX: line series for lower conformal band per spec
    QLineSeries *conformalUpperSeries;  // FIX: line series for upper conformal band per spec
};

#endif  // FIX: close header guard
