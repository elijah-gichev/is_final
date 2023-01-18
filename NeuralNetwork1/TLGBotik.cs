using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Extensions.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using System.Drawing;
using AForge.Imaging.Filters;

namespace NeuralNetwork1
{
    public class TLGBotik
    {
        public Telegram.Bot.TelegramBotClient botik = null;

        private UpdateTLGMessages formUpdater;

        private readonly AIMLService aiml;

        private BaseNetwork perseptron = null;
        // CancellationToken - инструмент для отмены задач, запущенных в отдельном потоке
        private readonly CancellationTokenSource cts = new CancellationTokenSource();
        public TLGBotik(BaseNetwork net,  UpdateTLGMessages updater, AIMLService aimlService)
        {
            aiml = aimlService;
            var botKey = System.IO.File.ReadAllText("botkey.txt");
            botik = new Telegram.Bot.TelegramBotClient(botKey);
            formUpdater = updater;
            perseptron = net;
        }

        public void SetNet(BaseNetwork net)
        {
            perseptron = net;
            formUpdater("Net updated!");
        }

        private async Task HandleUpdateMessageAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            //  Тут очень простое дело - банально отправляем назад сообщения
            var message = update.Message;
            formUpdater("Тип сообщения : " + message.Type.ToString());
            var chatId = message.Chat.Id;
            var username = message.Chat.FirstName;

            //  Получение файла (картинки)
            if (message.Type == Telegram.Bot.Types.Enums.MessageType.Photo)
            {
                formUpdater("Picture loadining started");
                var photoId = message.Photo.Last().FileId;
                Telegram.Bot.Types.File fl = botik.GetFileAsync(photoId).Result;
                var imageStream = new MemoryStream();
                await botik.DownloadFileAsync(fl.FilePath, imageStream, cancellationToken: cancellationToken);

                var img = System.Drawing.Image.FromStream(imageStream);
                Console.WriteLine("картинка получена");
                System.Drawing.Bitmap bm = new System.Drawing.Bitmap(img);
                
                Console.WriteLine("положили все в битмапу");

                ProcessImage(bm);
                ResizeBicubic resize = new ResizeBicubic(32, 32);
                AForge.Imaging.UnmanagedImage pr = resize.Apply(processed);
                Sample currentImage = new Sample(imgToData(pr), 10);

                Console.WriteLine("обработали изображение");

                Console.WriteLine("отмасштабировали изображение");

                FigureType recognizedClass = perseptron.Predict(currentImage);

                Console.WriteLine("сейчас будем угадывать");

                string talkabout = "";

                switch (recognizedClass)
                {
                    case FigureType.Play: talkabout = "play"; break;
                    case FigureType.Pause: talkabout = "pause"; break;
                    case FigureType.Repeat: talkabout = "repeat"; break;
                    case FigureType.Next: talkabout = "next"; break;
                    case FigureType.Previouse: talkabout = "previouse"; break;
                    case FigureType.Louder: talkabout = "louder"; break;
                    case FigureType.Quieter: talkabout = "quieter"; break;
                    case FigureType.Rewindf: talkabout = "rewindf"; break;
                    case FigureType.Rewindb: talkabout = "rewindb"; break;
                    case FigureType.Mix: talkabout = "mix"; break;
                    default: await botik.SendTextMessageAsync(message.Chat.Id, "Я такого не знаю!"); break;
                }

                string messageText = "Говорим о " + talkabout;
                string str = aiml.Talk(chatId, username, messageText);
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: str,
                    cancellationToken: cancellationToken);

                formUpdater("Picture recognized!");
                return;
            }

            if (message == null || message.Type != MessageType.Text) return;

            formUpdater("User: "+message.Text);

            if (message.Text.ToLower() == "что ты умеешь" || message.Text.ToLower() == "что ты умеешь?")
            {
                Bitmap bmCan = new Bitmap(@"C:\work\xxx\IS\NeuralNetrworkTLGBot_2021\NeuralNetrworkTLGBot\can\canBot.jpg");

                MemoryStream ms = new MemoryStream();
                bmCan.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                ms.Seek(0, 0);
                await botik.SendPhotoAsync(
                    message.Chat.Id,
                    ms,
                    //"Я умею распознавать эти символы",
                    cancellationToken: cancellationToken
                );
            }

            if (message.Type == MessageType.Text)
            {
                var messageText = update.Message.Text;

                string str = aiml.Talk(chatId, username, messageText);

                // Echo received message text
                await botClient.SendTextMessageAsync(
                    chatId: chatId,
                    text: str,
                    cancellationToken: cancellationToken);
                formUpdater("Bot: " + str);
                return;
            }

            return;
        }
        Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            var apiRequestException = exception as ApiRequestException;
            if (apiRequestException != null)
                Console.WriteLine($"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}");
            else
                Console.WriteLine(exception.ToString());
            return Task.CompletedTask;
        }

        public bool Act()
        {
            try
            {
                botik.StartReceiving(HandleUpdateMessageAsync, HandleErrorAsync, new ReceiverOptions
                {   // Подписываемся только на сообщения
                    AllowedUpdates = new[] { UpdateType.Message }
                },
                cancellationToken: cts.Token);
                // Пробуем получить логин бота - тестируем соединение и токен
                Console.WriteLine($"Connected as {botik.GetMeAsync().Result}");
            }
            catch(Exception e) { 
                return false;
            }
            return true;
        }

        public void Stop()
        {
            cts.Cancel();
        }
        //The class counts and extracts stand alone objects in images 
        AForge.Imaging.BlobCounter Blober = new AForge.Imaging.BlobCounter();
        
        public Bitmap original, final;
        public int BlobCount { get; private set; }
        public bool Recongnised { get; private set; }
        public float Angle { get; private set; }
        public float AngleRad { get; private set; }
        public float ThresholdValue = 0.2f;
        //The image processing routine implements local thresholding technique
        BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
        //The filter inverts colored and grayscale images.инверсия
        Invert InvertFilter = new AForge.Imaging.Filters.Invert();
        // filters objects
        Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
        // обработанное изображение 
        public AForge.Imaging.UnmanagedImage processed;

        public void ProcessImage(Bitmap input_image)
        {
            Blober.FilterBlobs = true;
            Blober.MinWidth = 5;
            Blober.MinHeight = 5;
            Blober.ObjectsOrder = AForge.Imaging.ObjectsOrder.Size;
            threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();

            InvertFilter = new AForge.Imaging.Filters.Invert();

            grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);

            int side = Math.Min(input_image.Height, input_image.Width);
            Rectangle cropRect = new Rectangle(0, 0, side, side);
            original = new Bitmap(input_image);                                                                                                                // set new processed
            if (processed != null)
                processed.Dispose();  //Конвертируем изображение в градации серого
            processed = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));
            //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
            threshldFilter.PixelBrightnessDifferenceLimit = ThresholdValue;
            threshldFilter.ApplyInPlace(processed);
            InvertFilter.ApplyInPlace(processed);
            Blober.ProcessImage(processed);
            AForge.Imaging.Blob[] blobs = Blober.GetObjectsInformation();
            BlobCount = blobs.Length;

            if (blobs.Length > 0)
            {
                var BiggestBlob = blobs[0];
                Recongnised = true;
                Blober.ExtractBlobsImage(processed, BiggestBlob, false);
                processed = BiggestBlob.Image;
                AForge.Point mc = BiggestBlob.CenterOfGravity;
                AForge.Point ic = new AForge.Point((float)BiggestBlob.Image.Width / 2, (float)BiggestBlob.Image.Height / 2);
                AngleRad = (ic.Y - mc.Y) / (ic.X - mc.X);
                Angle = (float)(Math.Atan(AngleRad) * 180 / Math.PI);
            }
            else
            {
                Recongnised = false;
                Angle = 0;
                AngleRad = -1;
            }
            if (final != null)
                final.Dispose();
            final = processed.ToManagedImage();
        }

        private double[] imgToData(AForge.Imaging.UnmanagedImage img)
        {
            double[] res = new double[img.Width * img.Height];
            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    //GetBrightness Возвращает значение освещенности (оттенок-насыщенность-освещенность (HSL)) для данной структуры
                    res[i * img.Width + j] = img.GetPixel(i, j).GetBrightness(); // maybe threshold
                }
            }
            return res;
        }
    }
}
