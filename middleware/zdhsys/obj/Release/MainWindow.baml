   M S B A M L   `   `   `  ���� O  KAI微纳实验软件, Version=1.0.0.0, Culture=neutral, PublicKeyToken=nullclr-namespace:zdhsyszdhsys  /clr-namespace:zdhsys.Controlzdhsys.Control      zdhsys.MainWindow   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     &localclr-namespace:zdhsys     /local1clr-namespace:zdhsys.Control     -     � WindowStyle$
  None=� � AllowsTransparency .  � Title$ 
MainWindow��$	��900��$
��1600�����	#FFF2F7FF � WindowStartupLocation$ CenterScreen=�.��� -    �Name$ Gd�� � ColumnDefinitions �� -    ��Name$ Cd_Left��$	��200 ��� $��* �.+��� $��0��
�������._�� -   $ Gd_Left��$��	20,0,20,0q� � RowDefinitions �� $	��100 ��� $��20 ��� $��60 ��� $��60 ��� $��60 ��� $��60 ��� $��6* �.+��� -    ��Name$ BtnMenu��$��Right=�$��0,30,0,0q�$��20��$��20������ .R��� $��列表收起.png���� $��1��$	��Top=�$��2�����#D8E9FF   zdhsys.Control.DevpButton  -   	  Name$	 Btn_home��   zdhsys.Control.DevpButton
   Click$
 Btn_home_Click��$��Center=�$	��150��$��40��$��2��  -   $	 
Btn_option��$
 Btn_home_Click��$��Center=�$	��150��$��40��$��3��  -   $	 
Btn_device��$
 Btn_home_Click��$��Center=�$	��150��$��40��$��4��  -	   $	 Btn_data��$
 Btn_home_Click��$��Center=�$	��150��$��40��$��5��� -
   $ Gd_Right��$��1�����#F2F7FF �� $��50 ��� $��* �.+�� $��0��
������� �� �� $��40 ��� $��40 ��� $��40 �.+�    zdhsys.Control.ImageButton  -     Name$ btn_min��    zdhsys.Control.ImageButton   Click$ Btn_Min_Click��$��1��$��50��  -   $ btn_max��$ Btn_Max_Click��$��2��$��50��  -   $ 	btn_close��$ Btn_Close_Click��$��3��$��50��� $��1��$��10q�.+��� -    ��Name$	 dcc��