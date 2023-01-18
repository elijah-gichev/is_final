using Accord.Neuro;
using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.Linq;
using System.Text;

using System.IO;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private string filename = "student_net_weights";

        // Входной взвешенный сигнал
        double[][] inputSignal;
        // Массив входных матриц весов
        double[][,] weights;
        // Значение ошибки
        double[][] errors;
        // Скорость обучения
        double learningConst = 0.0025;

        Stopwatch stopWatch = new Stopwatch();
        Random rand = new Random();
        int minWeight = -1;
        int maxWeight = 1;

        public StudentNetwork(int[] structure, double lowerBound = -1, double upperBound = 1)
        {
            inputSignal = new double[structure.Length][];
            errors = new double[structure.Length][];

            // инициализация начальных значений
            for (int i = 0; i < structure.Length; i++)
            {
                errors[i] = new double[structure[i]];
                inputSignal[i] = new double[structure[i] + 1];
                inputSignal[i][structure[i]] = 0;
            }

            weights = new double[structure.Length - 1][,]; // матрица весов

            // заполняем матрицу весов случайными значениями
            for (int n = 0; n < structure.Length - 1; n++)
            {
                int rowsCount = structure[n] + 1;
                int columnsCount = structure[n + 1];

                weights[n] = new double[rowsCount, columnsCount];

                for (int i = 0; i < rowsCount; i++)
                    for (int j = 0; j < columnsCount; j++)
                        weights[n][i, j] = minWeight + rand.NextDouble() * (maxWeight - minWeight);
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iteration = 1;
            bool f = iteration < 100;
            if (sample.error != null)
                f = sample.EstimatedError() > acceptableError;
            while (f)
            {
                Run(sample.input);

                // считаем ошибку по чудо-формуле
                for (var i = 0; i < sample.Output.Length; i++)
                {
                    double currentSignal = inputSignal[errors.Length - 1][i];
                    double expected = sample.Output[i];
                    errors[errors.Length - 1][i] = currentSignal * (1 - currentSignal) * (expected - currentSignal);
                }
                // перенос ошибки вглубь сети
                for (int i = errors.Length - 2; i >= 1; i--)
                {
                    for (int j = 0; j < errors[i].Length; j++)
                    {
                        double input = inputSignal[i][j] * (1 - inputSignal[i][j]);
                        double sum = 0.0;
                        for (int k = 0; k < errors[i + 1].Length; k++)
                            sum += errors[i + 1][k] * weights[i][j, k];
                        errors[i][j] = input * sum;
                    }
                }

                // корректирование матрицы ошибки
                for (int n = 0; n < weights.Length; n++)
                    for (int i = 0; i < weights[n].GetLength(0); i++)
                        for (int j = 0; j < weights[n].GetLength(1); j++)
                            weights[n][i, j] += learningConst * errors[n + 1][j] * inputSignal[n][i];

                iteration++;

                if (sample.error == null)
                    return iteration;
                f = sample.EstimatedError() > acceptableError;
            }

            return iteration;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            // Конструируем массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            // Группируем массивы из samplesSet в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            int epochToRun = 0;
            double samplesLooked = 0;
            double samplesCount = inputs.Length * epochsCount;
            double error = double.PositiveInfinity;

            stopWatch.Restart();

            while (epochToRun++ < epochsCount && error > acceptableError)
            {
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    Train(samplesSet[i], acceptableError, parallel);
                    error += EstimatedErrorFromOutput(outputs[i]);
                    samplesLooked++;
                }
                error /= inputs.Length;
                OnTrainProgress(samplesLooked / samplesCount, error, stopWatch.Elapsed);
            }

            OnTrainProgress(1, error, stopWatch.Elapsed);
            stopWatch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            Run(input);
            return inputSignal[inputSignal.Length - 1].Take(inputSignal.Last().Length - 1).ToArray();
        }

        private void Run(double[] input)
        {
            for (int j = 0; j < input.Length; j++)
                inputSignal[0][j] = input[j];

            for (int i = 1; i < inputSignal.GetLength(0); i++)
                Activate(inputSignal[i - 1], inputSignal[i], weights[i - 1]);
        }

        // Вычисление квадратичной ошибки по выходу сети, аналог sample.EstimatedError для массива
        private double EstimatedErrorFromOutput(double[] output)
        {
            double result = 0;

            for (int i = 0; i < output.Length; i++)
                result += Math.Pow(output[i] - inputSignal[inputSignal.Length - 1][i], 2);

            return result;
        }

        private static void Activate(double[] prevLauer, double[] layer, double[,] matrix)
        {
            int rowsCount = matrix.GetLength(0);
            int colCount = matrix.GetLength(1);

            for (int i = 0; i < colCount; i++)
            {
                double sum = 0;

                for (int j = 0; j < rowsCount; j++)
                    sum += prevLauer[j] * matrix[j, i];

                layer[i] = ActivateFunction(sum);
            }
        }

        private static double ActivateFunction(double x) => 1.0 / (Math.Exp(-x) + 1);

        public override void saveWeights()
        {
            string fName = System.IO.Path.Combine(Environment.CurrentDirectory, filename);
            File.WriteAllText(fName, JsonConvert.SerializeObject(weights, Formatting.Indented), Encoding.UTF8);
        }

        public override void loadWeights()
        {
            string fName = System.IO.Path.Combine(Environment.CurrentDirectory, filename);
            
            if (!File.Exists(fName))
            {
                MessageBox.Show("Нет файла для загрузки", "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                var jsonRes = File.ReadAllText(fName);
                weights = JsonConvert.DeserializeObject<double[][,]>(jsonRes);
            }
        }
    }
}