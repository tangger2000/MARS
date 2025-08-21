﻿using ICSharpCode.AvalonEdit.Document;
using ICSharpCode.AvalonEdit.Folding;
using System.Collections.Generic;

namespace JsonTool.Helper
{
    /// <summary>
	/// Allows producing foldings from a document based on braces.
	/// </summary>
	public class BraceFoldingStrategy
    {
        /// <summary>
        /// Gets/Sets the opening brace. The default value is '{'.
        /// </summary>
        public char OpeningBrace { get; set; }
        public char OpeningBrace2 { get; set; }
        /// <summary>
        /// Gets/Sets the closing brace. The default value is '}'.
        /// </summary>
        public char ClosingBrace { get; set; }
        public char ClosingBrace2 { get; set; }

        /// <summary>
        /// Creates a new BraceFoldingStrategy.
        /// </summary>
        public BraceFoldingStrategy()
        {
            this.OpeningBrace = '{';
            this.OpeningBrace2 = '[';
            this.ClosingBrace = '}';
            this.ClosingBrace2 = ']';
        }

        public void UpdateFoldings(FoldingManager manager, TextDocument document)
        {
            int firstErrorOffset;
            IEnumerable<NewFolding> newFoldings = CreateNewFoldings(document, out firstErrorOffset);
            manager.UpdateFoldings(newFoldings, firstErrorOffset);
        }

        /// <summary>
        /// Create <see cref="NewFolding"/>s for the specified document.
        /// </summary>
        public IEnumerable<NewFolding> CreateNewFoldings(TextDocument document, out int firstErrorOffset)
        {
            firstErrorOffset = -1;
            return CreateNewFoldings(document);
        }

        /// <summary>
        /// Create <see cref="NewFolding"/>s for the specified document.
        /// </summary>
        public IEnumerable<NewFolding> CreateNewFoldings(ITextSource document)
        {
            List<NewFolding> newFoldings = new List<NewFolding>();

            Stack<int> startOffsets = new Stack<int>();
            Stack<int> startOffsets2 = new Stack<int>();
            int lastNewLineOffset = 0;

            char openingBrace = this.OpeningBrace;
            char closingBrace = this.ClosingBrace;
            char openingBrace2 = this.OpeningBrace2;
            char closingBrace2 = this.ClosingBrace2;
            for (int i = 0; i < document.TextLength; i++)
            {
                char c = document.GetCharAt(i);
                if (c == openingBrace)
                {
                    startOffsets.Push(i);
                }
                else if (c == closingBrace && startOffsets.Count > 0)
                {
                    int startOffset = startOffsets.Pop();
                    // don't fold if opening and closing brace are on the same line
                    if (startOffset < lastNewLineOffset)
                    {
                        newFoldings.Add(new NewFolding(startOffset, i + 1));
                    }
                }
                else if (c == openingBrace2)
                {
                    startOffsets2.Push(i);
                }
                else if (c == closingBrace2 && startOffsets2.Count > 0)
                {
                    int startOffset = startOffsets2.Pop();
                    // don't fold if opening and closing brace are on the same line
                    if (startOffset < lastNewLineOffset)
                    {
                        newFoldings.Add(new NewFolding(startOffset, i + 1));
                    }
                }
                else if (c == '\n' || c == '\r')
                {
                    lastNewLineOffset = i + 1;
                }
            }
            newFoldings.Sort((a, b) => a.StartOffset.CompareTo(b.StartOffset));
            return newFoldings;
        }
    }
}
