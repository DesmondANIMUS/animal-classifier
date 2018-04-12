FROM ctava/tensorflow-go

RUN mkdir -p /model && \
    curl -o /model/inception5h.zip http://download.tensorflow.org/models/inception5h.zip && \
    unzip /model/inception5h.zip -d /model/

WORKDIR /go/src/imagerecog/
COPY . .
RUN go build

ENTRYPOINT ["/go/src/imagerecog/imagerecog"]