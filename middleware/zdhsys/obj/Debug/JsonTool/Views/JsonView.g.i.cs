﻿#pragma checksum "..\..\..\..\JsonTool\Views\JsonView.xaml" "{8829d00f-11b8-4213-878b-770e8597ac16}" "496F189D7EEF2274B201A4AACB3CBBB74613C16AA15C0F816B5C57A10AF1AB86"
//------------------------------------------------------------------------------
// <auto-generated>
//     此代码由工具生成。
//     运行时版本:4.0.30319.42000
//
//     对此文件的更改可能会导致不正确的行为，并且如果
//     重新生成代码，这些更改将会丢失。
// </auto-generated>
//------------------------------------------------------------------------------

using ICSharpCode.AvalonEdit;
using ICSharpCode.AvalonEdit.Editing;
using ICSharpCode.AvalonEdit.Highlighting;
using ICSharpCode.AvalonEdit.Rendering;
using ICSharpCode.AvalonEdit.Search;
using JsonTool.Behaviours;
using JsonTool.Controls;
using JsonTool.Converters;
using JsonTool.Converters.Json;
using Microsoft.Xaml.Behaviors;
using Microsoft.Xaml.Behaviors.Core;
using Microsoft.Xaml.Behaviors.Input;
using Microsoft.Xaml.Behaviors.Layout;
using Microsoft.Xaml.Behaviors.Media;
using Newtonsoft.Json.Linq;
using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;


namespace JsonTool.Views {
    
    
    /// <summary>
    /// JsonView
    /// </summary>
    public partial class JsonView : System.Windows.Controls.UserControl, System.Windows.Markup.IComponentConnector {
        
        
        #line 13 "..\..\..\..\JsonTool\Views\JsonView.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal JsonTool.Views.JsonView jsonView;
        
        #line default
        #line hidden
        
        
        #line 99 "..\..\..\..\JsonTool\Views\JsonView.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal JsonTool.Controls.AiButton btnFormat;
        
        #line default
        #line hidden
        
        
        #line 115 "..\..\..\..\JsonTool\Views\JsonView.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal ICSharpCode.AvalonEdit.TextEditor jsonEditor;
        
        #line default
        #line hidden
        
        private bool _contentLoaded;
        
        /// <summary>
        /// InitializeComponent
        /// </summary>
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        public void InitializeComponent() {
            if (_contentLoaded) {
                return;
            }
            _contentLoaded = true;
            System.Uri resourceLocater = new System.Uri("/AI微纳实验软件;component/jsontool/views/jsonview.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\..\..\JsonTool\Views\JsonView.xaml"
            System.Windows.Application.LoadComponent(this, resourceLocater);
            
            #line default
            #line hidden
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal System.Delegate _CreateDelegate(System.Type delegateType, string handler) {
            return System.Delegate.CreateDelegate(delegateType, this, handler);
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Design", "CA1033:InterfaceMethodsShouldBeCallableByChildTypes")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Maintainability", "CA1502:AvoidExcessiveComplexity")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1800:DoNotCastUnnecessarily")]
        void System.Windows.Markup.IComponentConnector.Connect(int connectionId, object target) {
            switch (connectionId)
            {
            case 1:
            this.jsonView = ((JsonTool.Views.JsonView)(target));
            
            #line 13 "..\..\..\..\JsonTool\Views\JsonView.xaml"
            this.jsonView.Loaded += new System.Windows.RoutedEventHandler(this.jsonView_Loaded);
            
            #line default
            #line hidden
            
            #line 13 "..\..\..\..\JsonTool\Views\JsonView.xaml"
            this.jsonView.Unloaded += new System.Windows.RoutedEventHandler(this.jsonView_Unloaded);
            
            #line default
            #line hidden
            return;
            case 2:
            this.btnFormat = ((JsonTool.Controls.AiButton)(target));
            return;
            case 3:
            this.jsonEditor = ((ICSharpCode.AvalonEdit.TextEditor)(target));
            
            #line 116 "..\..\..\..\JsonTool\Views\JsonView.xaml"
            this.jsonEditor.LostKeyboardFocus += new System.Windows.Input.KeyboardFocusChangedEventHandler(this.jsonEditor_LostKeyboardFocus);
            
            #line default
            #line hidden
            
            #line 117 "..\..\..\..\JsonTool\Views\JsonView.xaml"
            this.jsonEditor.LostFocus += new System.Windows.RoutedEventHandler(this.jsonEditor_LostFocus);
            
            #line default
            #line hidden
            return;
            }
            this._contentLoaded = true;
        }
    }
}

