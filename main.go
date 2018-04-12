package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sort"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	graphFile  = "/model/tensorflow_inception_graph.pb"
	labelsFile = "/model/imagenet_comp_graph_label_strings.txt"
)

type Label struct {
	Label       string
	Probability float32
}

type Labels []Label

func (l Labels) Len() int           { return len(l) }
func (l Labels) Swap(i, j int)      { l[i], l[j] = l[j], l[i] }
func (l Labels) Less(i, j int) bool { return l[i].Probability > l[j].Probability }

func handleErr(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: recog <img_url>")
	}
	fmt.Println("url: ", os.Args[1])

	resp := downloadImage(os.Args[1])
	defer resp.Body.Close()

	modelsGraph, labels := loadGraphLabels()

	session, err := tf.NewSession(modelsGraph, nil)
	handleErr(err)
	defer session.Close()

	tensor := normalizeImage(resp.Body)
	result, err := session.Run(map[tf.Output]*tf.Tensor{
		modelsGraph.Operation("input").Output(0): tensor,
	}, []tf.Output{
		modelsGraph.Operation("output").Output(0),
	}, nil)
	handleErr(err)

	topLabels := getTopLabels(labels, result[0].Value().([][]float32)[0])
	for _, l := range topLabels {
		fmt.Printf("Label: %s, Probability: %.2f%%\n", l.Label, l.Probability*100)
	}
}

func getTopLabels(labels []string, probabilities []float32) []Label {
	var results []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		results = append(results, Label{
			Label:       labels[i],
			Probability: p,
		})
	}
	sort.Sort(Labels(results))
	return results[:5]
}

func downloadImage(url string) *http.Response {
	resp, err := http.Get(url)
	handleErr(err)
	return resp
}

func loadGraphLabels() (*tf.Graph, []string) {
	model, err := ioutil.ReadFile(graphFile)
	handleErr(err)

	graph := tf.NewGraph()
	err = graph.Import(model, "")
	handleErr(err)

	f, err := os.Open(labelsFile)
	handleErr(err)
	defer f.Close()

	var labels []string

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return graph, labels
}

func normalizeImage(body io.ReadCloser) *tf.Tensor {
	var buf bytes.Buffer
	io.Copy(&buf, body)

	tensor, err := tf.NewTensor(buf.String())
	handleErr(err)

	graph, input, output := getNormalizedGraph()

	session, err := tf.NewSession(graph, nil)
	handleErr(err)
	defer session.Close()

	normalized, err := session.Run(map[tf.Output]*tf.Tensor{
		input: tensor,
	}, []tf.Output{
		output,
	}, nil)
	handleErr(err)

	return normalized[0]
}

func getNormalizedGraph() (*tf.Graph, tf.Output, tf.Output) {
	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	output := op.Sub(s,
		op.ResizeBilinear(s,
			op.ExpandDims(s,
				op.Cast(s, decode, tf.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		op.Const(s.SubScope("mean"), float32(117)))

	graph, err := s.Finalize()
	handleErr(err)

	return graph, input, output
}
