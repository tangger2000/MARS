   M S B A M L   `   `   `  ���� O  KAI微纳实验软件, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null+clr-namespace:zdhsys.Popupzdhsys.Popup  /clr-namespace:zdhsys.Controlzdhsys.Control  "    zdhsys.Popup.AddDeviceFields   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     ,localclr-namespace:zdhsys.Popup     /local1clr-namespace:zdhsys.Control     -     � WindowStyle$
  None=� � AllowsTransparency . 
������  � Title$ AddDeviceFields��$	��450��$	��800��.��� ���	#80000000.+�� $��Center=�$��Center=�$	��400��$	��230��
������� � RowDefinitions �� $��40 ��� $��10 ��� $��90 ��� $
��0.5* ��� $��1* �.+�� $��0�� � ColumnDefinitions �� $��30 ��� $��1* ��� $��1* �.+��� $��5��$��25�����#0066FF�� $��1��$��Center=�$��15(�$��添加设备可操作��$
��Bold"��� -    ��Name$ 	Btn_Close��$��Right=�$��0,0,10,0q�$��2��$��25��$��25��$��X��
�������
���  ��   NoBorderButtonStyle0#��[  �� $��1��$��2�����#E1E1E1$��	15,0,15,0q�$	��Top=�� $��2�� �� $��1* ��� $��4* � �� $��1* ��� $��1* �.+��� $��字段名称��$��Center=�  CommonStyleRight#��[ $
��Bold"�   zdhsys.Control.DevTextBox  -     Name$ 
txt_fields��$��1��$��10,0,0,0q�$	��250��$
��Left=��� $��内容��$��1��$��Center=�#��[ $
��Bold"�  -   $ txt_content��$��1��$��1��$��10,0,0,0q�$	��250��$
��Left=�� $��4�� �� $��2* ��� $��1* ��� $��1* ��� $
��0.2* �.+�   zdhsys.Control.DevButton0  -     Name$ Btn_Save��   zdhsys.Control.DevButton0   Click$ Btn_Save_Click��$��1��$��10,0,0,0q�$��40��  -   $ 
Btn_Close1��$ Btn_Close_Click��$��2��$��10,0,0,0q�$��40��