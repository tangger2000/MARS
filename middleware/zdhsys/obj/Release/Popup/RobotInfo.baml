   M S B A M L   `   `   `  ���� O  KAI微纳实验软件, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null+clr-namespace:zdhsys.Popupzdhsys.Popup  /clr-namespace:zdhsys.Controlzdhsys.Control      zdhsys.Popup.RobotInfo   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     ,localclr-namespace:zdhsys.Popup     /local1clr-namespace:zdhsys.Control     -     � WindowStyle$
  None=� � AllowsTransparency . 
������  � Title$ AddOptionInfo��$	��450��$	��800��.��� ���	#80000000.+�� $��Center=�$��Center=�$	��500��$	��200��
������� � RowDefinitions �� $��40 ��� $��10 ��� $��40 ��� $��40 ��� $��10 ��� $��40 �.+�� $��0�� � ColumnDefinitions �� $��30 ��� $��1* ��� $��1* �.+��� $��5��$��25�����#0066FF�� $��1��$��Center=�$��15(�$��机器人信息��$
��Bold"��� -    ��Name$ 	Btn_Close��$��Right=�$��0,0,10,0q�$��2��$��25��$��25��$��X��
�������
���  ��   NoBorderButtonStyle0#��[  �� $��1��$��2�����#E1E1E1$��	15,0,15,0q�$	��Top=�� $��2��$��	15,0,15,0q� �� $��1* ��� $��3* �.+��� $��IP地址��$��Center=�  CommonStyleRight#��[ $
��Bold"�   zdhsys.Control.DevTextBox  -     Name$ txt_IP��$��1��$��10,0,0,0q�$	��250��$
��Left=�� $��3��$��	15,0,15,0q� �� $��1* ��� $��3* �.+��� $��端口��$��Center=�#��[ $
��Bold"�  -   $ txt_Port��$��1��$��10,0,0,0q�$	��250��$
��Left=�� $��5�� �� $
��0.5* ��� $��1* ��� $��1* ��� $��1* ��� $
��0.5* �.+�   zdhsys.Control.DevButton0  -     Name$ Btn_Connect��   zdhsys.Control.DevButton0   Click$ Btn_Connect_Click��$��1��$��10,0,0,0q�$��40��  -   $ Btn_Save��$ Btn_Save_Click��$��2��$��10,0,0,0q�$��40��  -   $ 
Btn_Close1��$ Btn_Close_Click��$��3��$��10,0,0,0q�$��40��