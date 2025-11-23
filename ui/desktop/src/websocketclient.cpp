// FIX: WebSocketClient implementation for backend streaming connection per spec

#include "websocketclient.h"  // FIX: WebSocket client header

WebSocketClient::WebSocketClient(QObject *parent)  // FIX: constructor implementation
    : QObject(parent)  // FIX: call base constructor
    , webSocket(new QWebSocket("", QWebSocketProtocol::VersionLatest, this))  // FIX: create WebSocket instance
{
    // FIX: Connect WebSocket signals to handlers
    connect(webSocket, &QWebSocket::connected, this, &WebSocketClient::onConnected);  // FIX: connect connected signal
    connect(webSocket, &QWebSocket::disconnected, this, &WebSocketClient::onDisconnected);  // FIX: connect disconnected signal
    connect(webSocket, &QWebSocket::textMessageReceived, this, &WebSocketClient::onTextMessageReceived);  // FIX: connect message signal
    connect(webSocket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error), this, &WebSocketClient::onError);  // FIX: connect error signal
}

WebSocketClient::~WebSocketClient()  // FIX: destructor implementation
{
    // FIX: Close WebSocket connection
    if (webSocket && webSocket->isValid())  // FIX: check if WebSocket is valid
    {
        webSocket->close();  // FIX: close connection
    }
}

void WebSocketClient::connectToServer(const QString &url)  // FIX: connect to server implementation
{
    // FIX: Open WebSocket connection
    webSocket->open(QUrl(url));  // FIX: open connection
}

void WebSocketClient::disconnect()  // FIX: disconnect implementation
{
    // FIX: Close WebSocket connection
    if (webSocket && webSocket->isValid())  // FIX: check if WebSocket is valid
    {
        webSocket->close();  // FIX: close connection
    }
}

bool WebSocketClient::isConnected() const  // FIX: check connection status implementation
{
    return webSocket && webSocket->isValid();  // FIX: return connection status
}

void WebSocketClient::onConnected()  // FIX: connection handler implementation
{
    // FIX: Emit connected signal
    emit connected();  // FIX: emit signal
}

void WebSocketClient::onDisconnected()  // FIX: disconnection handler implementation
{
    // FIX: Emit disconnected signal
    emit disconnected();  // FIX: emit signal
}

void WebSocketClient::onTextMessageReceived(const QString &message)  // FIX: message handler implementation
{
    // FIX: Emit message received signal
    emit messageReceived(message);  // FIX: emit signal
}

void WebSocketClient::onError(QAbstractSocket::SocketError error)  // FIX: error handler implementation
{
    // FIX: Get error string
    QString errorString = webSocket->errorString();  // FIX: get error message

    // FIX: Emit error occurred signal
    emit errorOccurred(errorString);  // FIX: emit signal
}
