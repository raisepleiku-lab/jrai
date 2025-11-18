await query.edit_message_text(
            "Chọn coin để SHORT (1h):",
            reply_markup=build_short_menu_kb(),
        )

    elif data == "MENU_REPORT":
        await query.edit_message_text(
            "Chọn coin để xem report:",
            reply_markup=build_report_menu_kb(),
        )

    elif data == "SHOW_HELP":
        await context.bot.send_message(
            chat_id,
            get_help_text(),
            parse_mode=constants.ParseMode.MARKDOWN,
        )

    elif data.startswith("PLAN|"):
        try:
            _, side, sym, tf = data.split("|")
        except ValueError:
            await context.bot.send_message(chat_id, "Callback PLAN lỗi format.")
            return
        await suggest_plan(context, chat_id, sym, side, tf)

    elif data.startswith("REPORT|"):
        try:
            _, sym = data.split("|")
        except ValueError:
            await context.bot.send_message(chat_id, "Callback REPORT lỗi format.")
            return

        symbol = normalize_symbol(sym)
        text = build_full_report_text(symbol)
        await context.bot.send_message(
            chat_id,
            text,
            parse_mode=constants.ParseMode.MARKDOWN,
        )


# ========= MAIN =========
if name == "__main__":
    load_alerts()

    app = ApplicationBuilder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("report", report))

    app.add_handler(CommandHandler("longbtc", longbtc))
    app.add_handler(CommandHandler("shortbtc", shortbtc))
    app.add_handler(CommandHandler("longeth", longeth))
    app.add_handler(CommandHandler("shorteth", shorteth))
    app.add_handler(CommandHandler("longsol", longsol))
    app.add_handler(CommandHandler("shortsol", shortsol))
    app.add_handler(CommandHandler("longtrump", longtrump))
    app.add_handler(CommandHandler("shorttrump", shorttrump))

    app.add_handler(CommandHandler("alert", alert_cmd))

    # Inline callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))

    # JobQueue check alerts
    job = app.job_queue
    job.run_repeating(check_alerts, interval=60, first=10)

    print("Bot đang chạy… Ctrl+C để dừng.")
    app.run_polling()
