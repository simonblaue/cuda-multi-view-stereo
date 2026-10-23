/*
Copyright 2025 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "yaml_lite.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <utility>

namespace cv
{

namespace
{

std::string trim(const std::string& s)
{
	const size_t first = s.find_first_not_of(" \t\r\n");
	if (first == std::string::npos)
		return std::string();
	const size_t last = s.find_last_not_of(" \t\r\n");
	return s.substr(first, last - first + 1);
}

std::string unquote(const std::string& s)
{
	if (s.size() >= 2 &&
		((s.front() == '\"' && s.back() == '\"') || (s.front() == '\'' && s.back() == '\'')))
	{
		return s.substr(1, s.size() - 2);
	}
	return s;
}

int indentOf(const std::string& s)
{
	int indent = 0;
	for (const char c : s)
	{
		if (c == ' ')
			indent++;
		else if (c == '\t')
			indent += 4;
		else
			break;
	}
	return indent;
}

bool isBlank(const std::string& s)
{
	const std::string t = trim(s);
	return t.empty() || t[0] == '#';
}

std::string stripComment(const std::string& s)
{
	for (size_t i = 0; i < s.size(); i++)
	{
		// a '#' that starts a line or that follows a space starts a comment
		if (s[i] == '#' && (i == 0 || s[i - 1] == ' ' || s[i - 1] == '\t'))
			return s.substr(0, i);
	}
	return s;
}

/** @brief Finds the ':' that separates the key of an entry from its value. */
size_t findKeyColon(const std::string& s)
{
	int depth = 0;
	bool inString = false;
	char quote = '\0';

	for (size_t i = 0; i < s.size(); i++)
	{
		const char c = s[i];
		if (inString)
		{
			if (c == quote)
				inString = false;
			continue;
		}
		if (c == '\"' || c == '\'')
		{
			inString = true;
			quote = c;
			continue;
		}
		if (c == '[' || c == '{')
			depth++;
		else if (c == ']' || c == '}')
			depth--;
		else if (c == ':' && depth <= 0)
			return i;
	}
	return std::string::npos;
}

/** @brief Splits a string at the separators that are not nested. */
std::vector<std::string> splitTopLevel(const std::string& s, char separator)
{
	std::vector<std::string> parts;
	int depth = 0;
	bool inString = false;
	char quote = '\0';
	std::string current;

	for (const char c : s)
	{
		if (inString)
		{
			current += c;
			if (c == quote)
				inString = false;
			continue;
		}
		if (c == '\"' || c == '\'')
		{
			inString = true;
			quote = c;
			current += c;
			continue;
		}
		if (c == '[' || c == '{')
			depth++;
		else if (c == ']' || c == '}')
			depth--;

		if (c == separator && depth <= 0)
		{
			parts.push_back(current);
			current.clear();
			continue;
		}
		current += c;
	}
	parts.push_back(current);

	return parts;
}

/** @brief Splits the content of a flow mapping into its entries.
@param s Content of the mapping, without the enclosing braces.
@param closed Set to true if the end of the mapping was found.
@return The entries of the mapping.
*/
std::vector<std::string> splitFlowMap(const std::string& s, bool& closed)
{
	std::vector<std::string> parts;
	int depth = 0;
	bool inString = false;
	char quote = '\0';
	std::string current;
	closed = false;

	for (const char c : s)
	{
		if (inString)
		{
			current += c;
			if (c == quote)
				inString = false;
			continue;
		}
		if (c == '\"' || c == '\'')
		{
			inString = true;
			quote = c;
			current += c;
			continue;
		}
		if (c == '[' || c == '{')
		{
			depth++;
			current += c;
			continue;
		}
		if (c == ']' || c == '}')
		{
			if (c == '}' && depth == 0)
			{
				closed = true;
				break;
			}
			depth--;
			current += c;
			continue;
		}
		if (c == ',' && depth == 0)
		{
			parts.push_back(current);
			current.clear();
			continue;
		}
		current += c;
	}
	parts.push_back(current);

	return parts;
}

int bracketDepth(const std::string& s)
{
	int depth = 0;
	bool inString = false;
	char quote = '\0';

	for (const char c : s)
	{
		if (inString)
		{
			if (c == quote)
				inString = false;
			continue;
		}
		if (c == '\"' || c == '\'')
		{
			inString = true;
			quote = c;
			continue;
		}
		if (c == '[' || c == '{')
			depth++;
		else if (c == ']' || c == '}')
			depth--;
	}
	return depth;
}

bool isTag(const std::string& s)
{
	return s.size() > 2 && s[0] == '!' && s[1] == '!';
}

bool isMatrixField(const std::string& s)
{
	static const char* fields[] =
	{
		"rows:", "cols:", "dt:", "data:",
		"\"rows\":", "\"cols\":", "\"dt\":", "\"data\":"
	};
	for (const char* field : fields)
	{
		if (s.rfind(field, 0) == 0)
			return true;
	}
	return false;
}

int typeFromCode(char code)
{
	switch (code)
	{
	case 'd': return CV_64F;
	case 'f': return CV_32F;
	case 'i': return CV_32S;
	case 's': return CV_16S;
	case 'h': return CV_16U;
	case 'u': return CV_8U;
	case 'b': return CV_8S;
	default: return CV_64F;
	}
}

/** @brief The parser of the YAML subset. */
class YamlParser
{
public:
	explicit YamlParser(std::vector<std::string> lines) : lines_(std::move(lines)) {}

	YamlNode parse()
	{
		// skip the directives of the document
		while (index_ < lines_.size())
		{
			const std::string t = trim(lines_[index_]);
			if (t.empty() || t[0] == '#' || t == "---" || t == "..." ||
				t.rfind("%YAML", 0) == 0 || t.rfind("%TAG", 0) == 0)
			{
				index_++;
				continue;
			}
			break;
		}

		// a JSON document starts with an opening brace
		const std::string first = peekLine();
		if (!first.empty() && trim(first).front() == '{')
		{
			const std::string text = joinBalanced(trim(nextLine()));
			return parseFlowMap(text, 0);
		}

		return parseBlock(0);
	}

private:
	std::vector<std::string> lines_;
	size_t index_ = 0;
	std::string pending_;   //!< the remainder of a line that was read too early

	/** @brief Returns the next line of the document without consuming it. */
	std::string peekLine() const
	{
		if (!pending_.empty())
			return pending_;
		if (index_ >= lines_.size())
			return std::string();
		return lines_[index_];
	}

	/** @brief Consumes and returns the next line of the document. */
	std::string nextLine()
	{
		if (!pending_.empty())
		{
			const std::string text = pending_;
			pending_.clear();
			return text;
		}
		if (index_ >= lines_.size())
			return std::string();
		return lines_[index_++];
	}

	/** @brief Appends the following lines until the brackets are balanced.
	The characters that follow the balanced brackets are kept for the next
	caller, e.g. the closing brace of the enclosing flow mapping.
	@param text Text that is already read.
	@return The text with the balanced brackets.
	*/
	std::string joinBalanced(const std::string& text)
	{
		std::string result = text;
		while (bracketDepth(result) > 0)
		{
			const std::string next = nextLine();
			if (next.empty())
				break;

			int depth = bracketDepth(result);
			size_t consumed = 0;
			for (size_t i = 0; i < next.size(); i++)
			{
				const char c = next[i];
				if (c == '[' || c == '{')
					depth++;
				else if (c == ']' || c == '}')
					depth--;

				result += c;
				consumed = i + 1;

				if (depth <= 0)
					break;
			}
			pending_ = next.substr(consumed);

			if (depth <= 0)
				break;
		}

		return result;
	}

	/** @brief Parses a block mapping. */
	YamlNode parseBlock(int minIndent)
	{
		YamlNode node;
		node.type = YamlNode::MAP;

		while (true)
		{
			const std::string raw = peekLine();
			if (raw.empty())
				break;
			if (isBlank(raw))
			{
				nextLine();
				continue;
			}

			const int indent = indentOf(raw);
			if (indent < minIndent)
				break;

			const std::string line = trim(nextLine());
			const size_t colon = findKeyColon(line);
			if (colon == std::string::npos)
				continue;

			const std::string key = unquote(trim(line.substr(0, colon)));
			const std::string rest = trim(line.substr(colon + 1));

			node.map.emplace_back(key, parseValue(rest, indent));
		}

		return node;
	}

	/** @brief Parses the value of an entry. */
	YamlNode parseValue(const std::string& rest, int indent)
	{
		if (rest.empty())
		{
			// the value is a nested block
			return parseBlock(indent + 1);
		}
		if (rest.front() == '{')
		{
			return parseFlowMap(rest, indent);
		}
		if (rest.front() == '[')
		{
			return parseFlowSequence(joinBalanced(rest));
		}
		if (isTag(rest))
		{
			// the value is a block collection, e.g. an "!!opencv-matrix"
			return parseMatrixBlock();
		}
		return parseScalar(rest);
	}

	/** @brief Parses a flow mapping. */
	YamlNode parseFlowMap(const std::string& first, int baseIndent)
	{
		YamlNode node;
		node.type = YamlNode::MAP;

		std::string buffer = first;
		const size_t open = buffer.find('{');
		if (open != std::string::npos)
			buffer = buffer.substr(open + 1);

		while (true)
		{
			bool closed = false;
			const std::vector<std::string> entries = splitFlowMap(buffer, closed);
			for (const std::string& entry : entries)
			{
				const std::string item = trim(entry);
				if (item.empty())
					continue;

				const size_t colon = findKeyColon(item);
				if (colon == std::string::npos)
					continue;

				const std::string key = unquote(trim(item.substr(0, colon)));
				const std::string rest = trim(item.substr(colon + 1));
				node.map.emplace_back(key, parseValue(rest, baseIndent + 1));
			}

			if (closed)
				break;

			// the mapping continues on the next line
			const bool hadPending = !pending_.empty();
			const std::string raw = nextLine();
			if (raw.empty())
				break;
			if (!hadPending && indentOf(raw) < baseIndent)
				break;
			buffer = trim(raw);
		}

		return node;
	}

	/** @brief Parses a flow sequence. */
	YamlNode parseFlowSequence(const std::string& text)
	{
		YamlNode node;
		node.type = YamlNode::SEQUENCE;

		const size_t open = text.find('[');
		if (open == std::string::npos)
			return node;

		const size_t close = text.rfind(']');
		const size_t end = (close != std::string::npos && close > open) ? close : text.size();
		const std::string content = text.substr(open + 1, end - open - 1);

		for (const std::string& element : splitTopLevel(content, ','))
		{
			const std::string item = trim(element);
			if (item.empty())
				continue;
			if (item.front() == '{' || item.front() == '[')
			{
				node.sequence.push_back(parseValue(item, 0));
				continue;
			}
			node.sequence.push_back(parseScalar(item));
		}

		return node;
	}

	/** @brief Parses the block collection of an "!!opencv-matrix" node. */
	YamlNode parseMatrixBlock()
	{
		YamlNode node;
		node.type = YamlNode::MAP;

		while (true)
		{
			const std::string raw = peekLine();
			if (raw.empty())
				break;
			if (isBlank(raw))
			{
				nextLine();
				continue;
			}

			const std::string line = trim(raw);
			if (!isMatrixField(line))
				break;

			nextLine();

			const size_t colon = findKeyColon(line);
			if (colon == std::string::npos)
				break;

			const std::string key = unquote(trim(line.substr(0, colon)));
			const std::string rest = trim(line.substr(colon + 1));

			if (key == "data")
			{
				node.map.emplace_back(key, parseFlowSequence(joinBalanced(rest)));
			}
			else
			{
				node.map.emplace_back(key, parseScalar(rest));
			}
		}

		return node;
	}

	/** @brief Parses a scalar. */
	YamlNode parseScalar(const std::string& text)
	{
		YamlNode node;
		node.type = YamlNode::SCALAR;
		node.scalar = unquote(trim(text));
		return node;
	}
};

} // namespace

const YamlNode* YamlNode::find(const std::string& key) const
{
	for (const auto& entry : map)
	{
		if (entry.first == key)
			return &entry.second;
	}
	return nullptr;
}

std::string YamlNode::getString(const std::string& key, const std::string& defaultValue) const
{
	const YamlNode* node = find(key);
	if (node == nullptr || node->type != SCALAR)
		return defaultValue;
	return node->scalar;
}

double YamlNode::getDouble(const std::string& key, double defaultValue) const
{
	const std::string s = getString(key);
	if (s.empty())
		return defaultValue;
	try
	{
		return std::stod(s);
	}
	catch (...)
	{
		return defaultValue;
	}
}

int YamlNode::getInt(const std::string& key, int defaultValue) const
{
	const std::string s = getString(key);
	if (s.empty())
		return defaultValue;
	try
	{
		return std::stoi(s);
	}
	catch (...)
	{
		return defaultValue;
	}
}

Mat YamlNode::toMatrix() const
{
	const YamlNode* rowsNode = find("rows");
	const YamlNode* colsNode = find("cols");
	const YamlNode* dtNode = find("dt");
	const YamlNode* dataNode = find("data");
	if (rowsNode == nullptr || colsNode == nullptr || dtNode == nullptr || dataNode == nullptr)
		return Mat();

	const int rows = std::atoi(rowsNode->scalar.c_str());
	const int cols = std::atoi(colsNode->scalar.c_str());
	const int type = typeFromCode(dtNode->scalar.empty() ? 'd' : dtNode->scalar[0]);
	if (rows <= 0 || cols <= 0)
		return Mat();

	std::vector<double> values;
	values.reserve(dataNode->sequence.size());
	for (const YamlNode& element : dataNode->sequence)
		values.push_back(std::atof(element.scalar.c_str()));

	Mat matrix(rows, cols, type);
	const size_t count = std::min<size_t>(values.size(), static_cast<size_t>(rows) * cols);

	switch (type)
	{
	case CV_32F:
	{
		float* ptr = matrix.ptr<float>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = static_cast<float>(values[i]);
		break;
	}
	case CV_32S:
	{
		int* ptr = matrix.ptr<int>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = static_cast<int>(values[i]);
		break;
	}
	case CV_16S:
	{
		short* ptr = matrix.ptr<short>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = static_cast<short>(values[i]);
		break;
	}
	case CV_16U:
	{
		ushort* ptr = matrix.ptr<ushort>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = static_cast<ushort>(values[i]);
		break;
	}
	case CV_8U:
	{
		uchar* ptr = matrix.ptr<uchar>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = static_cast<uchar>(values[i]);
		break;
	}
	case CV_8S:
	{
		schar* ptr = matrix.ptr<schar>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = static_cast<schar>(values[i]);
		break;
	}
	default:
	{
		double* ptr = matrix.ptr<double>();
		for (size_t i = 0; i < count; i++)
			ptr[i] = values[i];
		break;
	}
	}

	return matrix;
}

Mat YamlNode::getMatrix(const std::string& key) const
{
	const YamlNode* node = find(key);
	if (node == nullptr)
		return Mat();
	return node->toMatrix();
}

YamlNode parseYamlFile(const String& filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
	{
		CV_Error(Error::StsBadArg,
			format("Cannot open the file '%s'", filename.c_str()).c_str());
	}

	std::vector<std::string> lines;
	std::string line;
	while (std::getline(file, line))
	{
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		lines.push_back(stripComment(line));
	}

	YamlParser parser(lines);
	return parser.parse();
}

} // namespace cv
