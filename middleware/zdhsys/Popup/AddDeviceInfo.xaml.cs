﻿using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using zdhsys.Bean;
using zdhsys.Control;
using zdhsys.entity;
using zdhsys.Unitils;

namespace zdhsys.Popup
{
    /// <summary>
    /// AddDeviceInfo.xaml 的交互逻辑
    /// </summary>
    public partial class AddDeviceInfo : Window
    {
        public AddDeviceInfo(double left, double top)
        {
            InitializeComponent();
            Left = left;
            Top = top;

            Btn_Add.Click += Btn_Add_Click;
            cbb_Group.SelectionChanged += Cbb_Group_SelectionChanged;
            Btn_Point.Click += Btn_Point_Click;
        }

        private void Btn_Point_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left, Top);
            AddDevicePoint dp = new AddDevicePoint(p)
            {
                Width = Width,
                Height = Height
            };
            if (!string.IsNullOrEmpty(Dim.PointJson))
            {
                dp.isUpdate = true;
                dp.dp = JsonConvert.DeserializeObject<DevicePoint>(Dim.PointJson);
            }
            _ = dp.ShowDialog();
            if (dp.flag_Save)
            {
                Dim.PointJson = JsonConvert.SerializeObject(dp.dp);
                Console.WriteLine($"PointJson={Dim.PointJson}");
            }
        }

        private void Cbb_Group_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            try
            {
                GlobalEnum.UnitDeviceType ut = (GlobalEnum.UnitDeviceType)cbb_Group.SelectedIndex;
                if (ut == GlobalEnum.UnitDeviceType.包硅反应设备)
                {
                    Btn_Point.Visibility = Visibility.Visible;
                }
                else
                {
                    Btn_Point.Visibility = Visibility.Hidden;
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        private List<DeviceFieldsModel> vs = new List<DeviceFieldsModel>();

        private void Btn_Add_Click(object sender, RoutedEventArgs e)
        {
            Point p = new Point(Left,Top);
            AddDeviceFields fields = new AddDeviceFields(p)
            {
                Width = Width,
                Height = Height
            };
            _ = fields.ShowDialog();
            if (fields.flag_save)
            {
                DeviceFieldsModel df = new DeviceFieldsModel
                {
                    FieldsContent = fields.fieldsContent,
                    FieldsName = fields.fieldsName
                };
                vs.Add(df);
                DataToUI();
            }
        }

        private void DataToUI()
        {
            gd_content.Children.Clear();
            gd_content.RowDefinitions.Clear();
            gd_content.ColumnDefinitions.Clear();

            for(int i = 0; i < vs.Count; i++)
            {
                DataToGridView(vs[i]);
            }
        }

        private void DataToGridView(DeviceFieldsModel df)
        {
            // 创建新的行- 固定行高:40
            GridViewUnitls.AddRow(gd_content);

            // 创建并添加新的列到新行
            for (int i = 0; i < gd_header.ColumnDefinitions.Count; i++)
            {
                //当前行，添加1列 -- 列宽对应表头
                GridViewUnitls.AddColumn(gd_content, gd_header, i);

                // 添加对应的控件到新的格子中
                if (i == 0)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.FieldsName);
                }
                else if (i == 1)
                {
                    GridViewUnitls.AddCell(gd_content, i, df.FieldsContent);
                }
                else if (i == 2)
                {
                    DevOperation2 dev = new DevOperation2();
                    dev.HorizontalAlignment = HorizontalAlignment.Center;
                    dev.VerticalAlignment = VerticalAlignment.Center;
                    dev.obj = df;//绑定对象。用于详情，修改，删除之类的。
                    dev.Click_Update += Dev_Click_Update;
                    dev.Click_Delete += Dev_Click_Delete;
                    Grid.SetRow(dev, gd_content.RowDefinitions.Count - 1);
                    Grid.SetColumn(dev, i);
                    gd_content.Children.Add(dev);
                }
            }
            //每行再加一个分割线
            GridViewUnitls.AddSpLine(gd_content, gd_header);
        }

        private void Dev_Click_Delete(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            DeviceFieldsModel df = dev.obj as DeviceFieldsModel;
            for (int i = 0; i < vs.Count; i++)
            {
                if (vs[i].Id == df.Id && vs[i].FieldsName == df.FieldsName && vs[i].FieldsContent == df.FieldsContent)
                {
                    vs.RemoveAt(i);
                    break;
                }
            }
            DataToUI();
        }

        private void Dev_Click_Update(object sender, RoutedEventArgs e)
        {
            DevOperation2 dev = sender as DevOperation2;
            DeviceFieldsModel df = dev.obj as DeviceFieldsModel;
            int index = -1;
            for (int i = 0; i < vs.Count; i++)
            {
                if (vs[i].Id == df.Id && vs[i].FieldsName == df.FieldsName && vs[i].FieldsContent == df.FieldsContent)
                {
                    index = i;
                    break;
                }
            }

            Point p = new Point(Left, Top);
            AddDeviceFields fields = new AddDeviceFields(p)
            {
                Width = Width,
                Height = Height,
                fieldsContent = vs[index].FieldsContent,
                fieldsName = vs[index].FieldsName
            };
            _ = fields.ShowDialog();
            if (fields.flag_save)
            {
                vs[index].FieldsContent = fields.fieldsContent;
                vs[index].FieldsName = fields.fieldsName;
                DataToUI();
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            Btn_Save.SetUI("#0066FF", "#0066FF", "#FFFFFF", "保存");
            Btn_Add.SetUI("#0066FF", "#0066FF", "#FFFFFF", "添加");
            Btn_Point.SetUI("#0066FF", "#0066FF", "#FFFFFF", "点位");
            Btn_Point.Visibility = Visibility.Hidden;
            Btn_Close1.SetUI("#FFFFFF", "#0066FF", "#0066FF", "取消");

            txt_DeviceID.SetUI("输入设备ID");
            txt_DeviceName.SetUI("输入设备名称");
            txt_TagName.SetUI("输入标签名称");

            txt_x.SetUI("输入坐标数值");
            txt_x.SetNumber();
            txt_y.SetUI("输入坐标数值");
            txt_y.SetNumber();
            txt_z.SetUI("输入坐标数值");
            txt_z.SetNumber();

            txt_len.SetUI("输入长");
            txt_len.SetNumber();
            txt_w.SetUI("输入宽");
            txt_w.SetNumber();
            txt_h.SetUI("输入高");
            txt_h.SetNumber();


            foreach (GlobalEnum.UnitOfMeasurement unit in Enum.GetValues(typeof(GlobalEnum.UnitOfMeasurement)))
            {
                cbb_Tag.Items.Add(unit);
            }
            cbb_Tag.SelectedIndex = 0;

            cbb_Bottle.Items.Add(GlobalEnum.UnitBottle.大瓶子);
            cbb_Bottle.Items.Add(GlobalEnum.UnitBottle.小瓶子);
            cbb_Bottle.SelectedIndex = 0;

            foreach (GlobalEnum.UnitDeviceType unit in Enum.GetValues(typeof(GlobalEnum.UnitDeviceType)))
            {
                cbb_Group.Items.Add(unit);
            }
            cbb_Group.SelectedIndex = 0;

            if (isUpdate)
            {
                txt_DeviceID.SetText(Dim.DeviceId.ToString());
                txt_DeviceName.SetText(Dim.DeviceName);
                txt_TagName.SetText(Dim.TagName);
                cbb_Tag.Text = Dim.TagUnit;
                txt_x.SetText(Dim.X.ToString());
                txt_y.SetText(Dim.Y.ToString());
                txt_z.SetText(Dim.Z.ToString());
                txt_len.SetText(Dim.L.ToString());
                txt_w.SetText(Dim.W.ToString());
                txt_h.SetText(Dim.H.ToString());
                cbb_Bottle.SelectedIndex = Dim.Bottole;
                if (!string.IsNullOrEmpty(Dim.DeviceGroupId))
                {
                    cbb_Group.SelectedIndex = int.Parse(Dim.DeviceGroupId);  
                }
                vs = Dim.Dfms;
                DataToUI();
            }
            else
            {
                Dim = new DeviceInfoModel();
            }
        }


        public DeviceInfoModel Dim;
        /// <summary>
        /// 0反应类设备 1转移类设备
        /// </summary>
        public int DeviceType = 0;
        public bool isUpdate = false;
        public bool flag_save = false;
        /// <summary>
        /// 保存设备信息--区分新增和修改
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_Save_Click(object sender, RoutedEventArgs e)
        {
            //是否新增
            if (!isUpdate)
            {
                //这里新增最大排序号来。不按记录条数。因为记录会有删除的可能
                List<DeviceInfoModel> vs = SqlHelper.GetDeviceInfo();
                int index = 1;
                if (vs.Count > 0)
                {
                    index = vs[vs.Count - 1].Sort + 1;
                }
                Dim.DeviceType = DeviceType;
                Dim.CreateTime = GlobalUnitils.GetNowTime(DateTime.Now);
                Dim.Sort = index;
            }
            Dim.DeviceId = txt_DeviceID.Text();
            Dim.DeviceName = txt_DeviceName.Text();
            Dim.TagName = txt_TagName.Text();
            Dim.TagUnit = cbb_Tag.Text;
            Dim.Bottole = cbb_Bottle.SelectedIndex;
            Dim.X = float.Parse(string.IsNullOrEmpty(txt_x.Text()) ? "0" : txt_x.Text());
            Dim.Y = float.Parse(string.IsNullOrEmpty(txt_y.Text()) ? "0" : txt_y.Text());
            Dim.Z = float.Parse(string.IsNullOrEmpty(txt_z.Text()) ? "0" : txt_z.Text());
            Dim.L = float.Parse(string.IsNullOrEmpty(txt_len.Text()) ? "0" : txt_len.Text());
            Dim.W = float.Parse(string.IsNullOrEmpty(txt_w.Text()) ? "0" : txt_w.Text());
            Dim.H = float.Parse(string.IsNullOrEmpty(txt_h.Text()) ? "0" : txt_h.Text());
            // 设备类型
            Dim.DeviceGroupId = cbb_Group.SelectedIndex.ToString();
            for (int i = 0; i < vs.Count; i++)
            {
                vs[i].Id = i;
            }
            Dim.Dfms = vs;
            Dim.FieldJson = JsonConvert.SerializeObject(vs);
            flag_save = true;
            Close();
        }

        private void Btn_Close_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}
