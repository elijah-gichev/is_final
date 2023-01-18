namespace NeuralNetwork1
{
    partial class Form1
    {
        /// <summary>
        /// Обязательная переменная конструктора.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Освободить все используемые ресурсы.
        /// </summary>
        /// <param name="disposing">истинно, если управляемый ресурс должен быть удален; иначе ложно.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Код, автоматически созданный конструктором форм Windows

        /// <summary>
        /// Требуемый метод для поддержки конструктора — не изменяйте 
        /// содержимое этого метода с помощью редактора кода.
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.Label label2;
            System.Windows.Forms.Label label5;
            System.Windows.Forms.Label label6;
            System.Windows.Forms.Label label7;
            this.label1 = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.label11 = new System.Windows.Forms.Label();
            this.netTypeBox = new System.Windows.Forms.ComboBox();
            this.parallelCheckBox = new System.Windows.Forms.CheckBox();
            this.netStructureBox = new System.Windows.Forms.TextBox();
            this.recreateNetButton = new System.Windows.Forms.Button();
            this.netTrainButton = new System.Windows.Forms.Button();
            this.AccuracyCounter = new System.Windows.Forms.TrackBar();
            this.EpochesCounter = new System.Windows.Forms.NumericUpDown();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.trainOneButton = new System.Windows.Forms.Button();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.infoStatusLabel = new System.Windows.Forms.ToolStripStatusLabel();
            this.elapsedTimeLabel = new System.Windows.Forms.Label();
            this.вапрвапрToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.StatusLabel = new System.Windows.Forms.Label();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.TLGBotOnButton = new System.Windows.Forms.Button();
            this.TLGUsersMessages = new System.Windows.Forms.TextBox();
            this.button2 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            label2 = new System.Windows.Forms.Label();
            label5 = new System.Windows.Forms.Label();
            label6 = new System.Windows.Forms.Label();
            label7 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.AccuracyCounter)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.EpochesCounter)).BeginInit();
            this.statusStrip1.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.SuspendLayout();
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new System.Drawing.Point(103, 58);
            label2.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            label2.Name = "label2";
            label2.Size = new System.Drawing.Size(110, 16);
            label2.TabIndex = 2;
            label2.Text = "Структура сети";
            // 
            // label5
            // 
            label5.AutoSize = true;
            label5.Location = new System.Drawing.Point(92, 123);
            label5.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            label5.Name = "label5";
            label5.Size = new System.Drawing.Size(118, 16);
            label5.TabIndex = 7;
            label5.Text = "Количество эпох";
            // 
            // label6
            // 
            label6.AutoSize = true;
            label6.Location = new System.Drawing.Point(31, 239);
            label6.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            label6.Name = "label6";
            label6.Size = new System.Drawing.Size(69, 16);
            label6.TabIndex = 9;
            label6.Text = "Точность";
            // 
            // label7
            // 
            label7.AutoSize = true;
            label7.Location = new System.Drawing.Point(46, 425);
            label7.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            label7.Name = "label7";
            label7.Size = new System.Drawing.Size(47, 16);
            label7.TabIndex = 14;
            label7.Text = "Status:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 15);
            this.label1.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(104, 16);
            this.label1.TabIndex = 2;
            this.label1.Text = "Какой-то текст";
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.button3);
            this.groupBox1.Controls.Add(this.button2);
            this.groupBox1.Controls.Add(this.label11);
            this.groupBox1.Controls.Add(this.netTypeBox);
            this.groupBox1.Controls.Add(this.parallelCheckBox);
            this.groupBox1.Controls.Add(this.netStructureBox);
            this.groupBox1.Controls.Add(this.recreateNetButton);
            this.groupBox1.Controls.Add(this.netTrainButton);
            this.groupBox1.Controls.Add(this.AccuracyCounter);
            this.groupBox1.Controls.Add(label6);
            this.groupBox1.Controls.Add(this.EpochesCounter);
            this.groupBox1.Controls.Add(label5);
            this.groupBox1.Controls.Add(label2);
            this.groupBox1.Location = new System.Drawing.Point(13, 43);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox1.Size = new System.Drawing.Size(392, 368);
            this.groupBox1.TabIndex = 3;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Параметры сети";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(172, 27);
            this.label11.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(38, 16);
            this.label11.TabIndex = 21;
            this.label11.Text = "Сеть";
            // 
            // netTypeBox
            // 
            this.netTypeBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.netTypeBox.FormattingEnabled = true;
            this.netTypeBox.Location = new System.Drawing.Point(221, 23);
            this.netTypeBox.Margin = new System.Windows.Forms.Padding(4);
            this.netTypeBox.Name = "netTypeBox";
            this.netTypeBox.Size = new System.Drawing.Size(160, 24);
            this.netTypeBox.TabIndex = 20;
            this.netTypeBox.SelectedIndexChanged += new System.EventHandler(this.netTypeBox_SelectedIndexChanged);
            // 
            // parallelCheckBox
            // 
            this.parallelCheckBox.AutoSize = true;
            this.parallelCheckBox.Checked = true;
            this.parallelCheckBox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.parallelCheckBox.Location = new System.Drawing.Point(45, 299);
            this.parallelCheckBox.Margin = new System.Windows.Forms.Padding(4);
            this.parallelCheckBox.Name = "parallelCheckBox";
            this.parallelCheckBox.Size = new System.Drawing.Size(176, 20);
            this.parallelCheckBox.TabIndex = 19;
            this.parallelCheckBox.Text = "Параллельный расчёт";
            this.parallelCheckBox.UseVisualStyleBackColor = true;
            // 
            // netStructureBox
            // 
            this.netStructureBox.Location = new System.Drawing.Point(223, 54);
            this.netStructureBox.Margin = new System.Windows.Forms.Padding(4);
            this.netStructureBox.Name = "netStructureBox";
            this.netStructureBox.Size = new System.Drawing.Size(159, 22);
            this.netStructureBox.TabIndex = 18;
            this.netStructureBox.Text = "1024;800;150;10";
            // 
            // recreateNetButton
            // 
            this.recreateNetButton.Location = new System.Drawing.Point(103, 198);
            this.recreateNetButton.Margin = new System.Windows.Forms.Padding(4);
            this.recreateNetButton.Name = "recreateNetButton";
            this.recreateNetButton.Size = new System.Drawing.Size(187, 37);
            this.recreateNetButton.TabIndex = 17;
            this.recreateNetButton.Text = "Пересоздать сеть";
            this.recreateNetButton.UseVisualStyleBackColor = true;
            this.recreateNetButton.Click += new System.EventHandler(this.button3_Click);
            this.recreateNetButton.MouseEnter += new System.EventHandler(this.recreateNetButton_MouseEnter);
            // 
            // netTrainButton
            // 
            this.netTrainButton.Location = new System.Drawing.Point(3, 323);
            this.netTrainButton.Margin = new System.Windows.Forms.Padding(4);
            this.netTrainButton.Name = "netTrainButton";
            this.netTrainButton.Size = new System.Drawing.Size(133, 37);
            this.netTrainButton.TabIndex = 11;
            this.netTrainButton.Text = "Обучить";
            this.netTrainButton.UseVisualStyleBackColor = true;
            this.netTrainButton.Click += new System.EventHandler(this.button1_Click);
            this.netTrainButton.MouseEnter += new System.EventHandler(this.netTrainButton_MouseEnter);
            // 
            // AccuracyCounter
            // 
            this.AccuracyCounter.Location = new System.Drawing.Point(33, 258);
            this.AccuracyCounter.Margin = new System.Windows.Forms.Padding(4);
            this.AccuracyCounter.Maximum = 100;
            this.AccuracyCounter.Name = "AccuracyCounter";
            this.AccuracyCounter.Size = new System.Drawing.Size(327, 56);
            this.AccuracyCounter.TabIndex = 10;
            this.AccuracyCounter.TickFrequency = 10;
            this.AccuracyCounter.Value = 80;
            // 
            // EpochesCounter
            // 
            this.EpochesCounter.Location = new System.Drawing.Point(223, 121);
            this.EpochesCounter.Margin = new System.Windows.Forms.Padding(4);
            this.EpochesCounter.Maximum = new decimal(new int[] {
            1000000,
            0,
            0,
            0});
            this.EpochesCounter.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.EpochesCounter.Name = "EpochesCounter";
            this.EpochesCounter.Size = new System.Drawing.Size(160, 22);
            this.EpochesCounter.TabIndex = 8;
            this.EpochesCounter.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(60, 456);
            this.label8.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(44, 16);
            this.label8.TabIndex = 6;
            this.label8.Text = "label8";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(15, 456);
            this.label9.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(17, 64);
            this.label9.TabIndex = 7;
            this.label9.Text = "0:\r\n1:\r\n2:\r\n3:";
            // 
            // trainOneButton
            // 
            this.trainOneButton.Location = new System.Drawing.Point(4, 595);
            this.trainOneButton.Margin = new System.Windows.Forms.Padding(4);
            this.trainOneButton.Name = "trainOneButton";
            this.trainOneButton.Size = new System.Drawing.Size(168, 37);
            this.trainOneButton.TabIndex = 8;
            this.trainOneButton.Text = "Обучить образцу";
            this.trainOneButton.UseVisualStyleBackColor = true;
            // 
            // progressBar1
            // 
            this.progressBar1.Location = new System.Drawing.Point(8, 526);
            this.progressBar1.Margin = new System.Windows.Forms.Padding(4);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(388, 27);
            this.progressBar1.Step = 1;
            this.progressBar1.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.progressBar1.TabIndex = 10;
            // 
            // statusStrip1
            // 
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.infoStatusLabel});
            this.statusStrip1.Location = new System.Drawing.Point(0, 640);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Padding = new System.Windows.Forms.Padding(1, 0, 19, 0);
            this.statusStrip1.Size = new System.Drawing.Size(944, 26);
            this.statusStrip1.TabIndex = 11;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // infoStatusLabel
            // 
            this.infoStatusLabel.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.infoStatusLabel.Name = "infoStatusLabel";
            this.infoStatusLabel.Size = new System.Drawing.Size(100, 20);
            this.infoStatusLabel.Text = "Обучите сеть";
            this.infoStatusLabel.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // elapsedTimeLabel
            // 
            this.elapsedTimeLabel.AutoSize = true;
            this.elapsedTimeLabel.Location = new System.Drawing.Point(15, 557);
            this.elapsedTimeLabel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.elapsedTimeLabel.Name = "elapsedTimeLabel";
            this.elapsedTimeLabel.Size = new System.Drawing.Size(51, 16);
            this.elapsedTimeLabel.TabIndex = 12;
            this.elapsedTimeLabel.Text = "Время:";
            // 
            // вапрвапрToolStripMenuItem
            // 
            this.вапрвапрToolStripMenuItem.Name = "вапрвапрToolStripMenuItem";
            this.вапрвапрToolStripMenuItem.Size = new System.Drawing.Size(180, 22);
            this.вапрвапрToolStripMenuItem.Text = "вапрвапр";
            // 
            // StatusLabel
            // 
            this.StatusLabel.AutoSize = true;
            this.StatusLabel.Location = new System.Drawing.Point(103, 425);
            this.StatusLabel.Margin = new System.Windows.Forms.Padding(4, 0, 4, 0);
            this.StatusLabel.Name = "StatusLabel";
            this.StatusLabel.Size = new System.Drawing.Size(46, 16);
            this.StatusLabel.TabIndex = 15;
            this.StatusLabel.Text = "NONE";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.TLGBotOnButton);
            this.groupBox3.Controls.Add(this.TLGUsersMessages);
            this.groupBox3.Location = new System.Drawing.Point(444, 256);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(4);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(4);
            this.groupBox3.Size = new System.Drawing.Size(475, 380);
            this.groupBox3.TabIndex = 17;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Telegram";
            this.groupBox3.Enter += new System.EventHandler(this.groupBox3_Enter);
            // 
            // TLGBotOnButton
            // 
            this.TLGBotOnButton.Location = new System.Drawing.Point(292, 23);
            this.TLGBotOnButton.Margin = new System.Windows.Forms.Padding(4);
            this.TLGBotOnButton.Name = "TLGBotOnButton";
            this.TLGBotOnButton.Size = new System.Drawing.Size(163, 42);
            this.TLGBotOnButton.TabIndex = 1;
            this.TLGBotOnButton.Text = "BotOn";
            this.TLGBotOnButton.UseVisualStyleBackColor = true;
            this.TLGBotOnButton.Click += new System.EventHandler(this.TLGBotOnButton_Click);
            // 
            // TLGUsersMessages
            // 
            this.TLGUsersMessages.Location = new System.Drawing.Point(8, 81);
            this.TLGUsersMessages.Margin = new System.Windows.Forms.Padding(4);
            this.TLGUsersMessages.Multiline = true;
            this.TLGUsersMessages.Name = "TLGUsersMessages";
            this.TLGUsersMessages.Size = new System.Drawing.Size(457, 291);
            this.TLGUsersMessages.TabIndex = 0;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(171, 323);
            this.button2.Margin = new System.Windows.Forms.Padding(4);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(101, 37);
            this.button2.TabIndex = 22;
            this.button2.Text = "Сохранить";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(280, 322);
            this.button3.Margin = new System.Windows.Forms.Padding(4);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(101, 37);
            this.button3.TabIndex = 23;
            this.button3.Text = "Загрузить";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click_1);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(944, 666);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.StatusLabel);
            this.Controls.Add(label7);
            this.Controls.Add(this.elapsedTimeLabel);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.progressBar1);
            this.Controls.Add(this.trainOneButton);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.MaximizeBox = false;
            this.Name = "Form1";
            this.Text = "Банальный студенческий перспетрон";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.AccuracyCounter)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.EpochesCounter)).EndInit();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.NumericUpDown EpochesCounter;
        private System.Windows.Forms.TrackBar AccuracyCounter;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Button recreateNetButton;
        private System.Windows.Forms.Button trainOneButton;
        private System.Windows.Forms.TextBox netStructureBox;
		private System.Windows.Forms.ProgressBar progressBar1;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel infoStatusLabel;
        private System.Windows.Forms.Label elapsedTimeLabel;
        private System.Windows.Forms.Button netTrainButton;
        private System.Windows.Forms.CheckBox parallelCheckBox;
        private System.Windows.Forms.ToolStripMenuItem вапрвапрToolStripMenuItem;
        private System.Windows.Forms.ComboBox netTypeBox;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.Label StatusLabel;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Button TLGBotOnButton;
        private System.Windows.Forms.TextBox TLGUsersMessages;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button2;
    }
}

